"""
SAM3 Video Tracking Nodes for ComfyUI
Memory-Safe & Unified Editor Edition + Direct File Preview API
"""

import os
import json
import base64
import hashlib
import logging
import tempfile
import shutil
import asyncio
from io import BytesIO
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

from aiohttp import web
from server import PromptServer

import folder_paths
from .utils import get_comfy_models_dir
from .sam3_lib.model_builder import build_sam3_video_predictor

log = logging.getLogger("sam3.video")

# =============================================================================
# Global RAM caches
# =============================================================================
_VIDEO_MODEL_CACHE = {}       # {ckpt_path: predictor}
_SESSION_FRAMES_CACHE = {}    # {session_id: {"b64":[...], "w":W, "h":H, "n":N}}
_PREVIEW_CACHE = {}           # {preview_key: {"b64":[...], "w":..., "h":..., "n":...}}


# =============================================================================
# Utility functions
# =============================================================================
def _frames_signature(video_frames):
    if video_frames is None or not hasattr(video_frames, "shape"):
        return "none"
    n = int(video_frames.shape[0])
    h = int(video_frames.shape[1])
    w = int(video_frames.shape[2])
    try:
        sub_0 = video_frames[0, ::32, ::32].contiguous().cpu().numpy()
        sub_m = video_frames[n // 2, ::32, ::32].contiguous().cpu().numpy()
        sub_l = video_frames[n - 1, ::32, ::32].contiguous().cpu().numpy()
        val_hash = hashlib.md5(sub_0.tobytes() + sub_m.tobytes() + sub_l.tobytes()).hexdigest()
    except Exception:
        val_hash = "error"
    return f"{n}x{h}x{w}_{val_hash}"


def _encode_frames_for_ui(frames, max_w=640, q=75):
    out = []
    for i in range(int(frames.shape[0])):
        arr = (np.clip(frames[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if pil.width > max_w:
            pil = pil.resize((max_w, int(pil.height * max_w / pil.width)), Image.BILINEAR)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _extract_frames_from_file(filename, max_w=640, max_frames=150, q=75):
    """Directly extracts preview frames from video/image file path before Queue execution."""
    file_path = None
    try:
        file_path = folder_paths.get_annotated_filepath(filename)
    except Exception:
        pass

    if not file_path or not os.path.exists(file_path):
        p = Path(filename)
        if p.exists():
            file_path = str(p)
        else:
            inp_dir = Path(folder_paths.get_input_directory())
            cand = inp_dir / filename
            if cand.exists():
                file_path = str(cand)

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Video/Image file not found: {filename}")

    out_b64 = []
    w, h = 0, 0

    # 1. Try Image / GIF / Animated WebP
    try:
        im = Image.open(file_path)
        if getattr(im, "is_animated", False):
            n_frames = min(im.n_frames, max_frames)
            for i in range(n_frames):
                im.seek(i)
                frame = im.convert("RGB")
                if i == 0:
                    w, h = frame.width, frame.height
                if frame.width > max_w:
                    frame = frame.resize((max_w, int(frame.height * max_w / frame.width)), Image.BILINEAR)
                buf = BytesIO()
                frame.save(buf, format="JPEG", quality=q)
                out_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            return out_b64, w, h
        else:
            frame = im.convert("RGB")
            w, h = frame.width, frame.height
            if frame.width > max_w:
                frame = frame.resize((max_w, int(frame.height * max_w / frame.width)), Image.BILINEAR)
            buf = BytesIO()
            frame.save(buf, format="JPEG", quality=q)
            out_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            return out_b64, w, h
    except Exception:
        pass

    # 2. Try Video via OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                if pil_img.width > max_w:
                    pil_img = pil_img.resize((max_w, int(pil_img.height * max_w / pil_img.width)), Image.BILINEAR)
                buf = BytesIO()
                pil_img.save(buf, format="JPEG", quality=q)
                out_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                count += 1
            cap.release()
            if out_b64:
                return out_b64, w, h
    except Exception as cv_err:
        log.warning(f"cv2 frame extraction failed: {cv_err}")

    raise RuntimeError(f"Could not read video/image frames from: {file_path}")


def _session_alive(video_model, session_id):
    if not session_id:
        return False
    try:
        getter = getattr(video_model, "_get_session", None)
        if getter is None:
            return True
        getter(session_id)
        return True
    except Exception:
        return False


def _recover_session(session, skip_node=None):
    video_model = session["model"]
    frames = session.get("frames")
    if frames is None:
        raise RuntimeError("[SAM3] Cannot recover session: session['frames'] missing.")

    _ensure_video_model_on_device(video_model)

    old_session_id = session.get("session_id")
    if old_session_id:
        try:
            video_model.close_session(old_session_id)
        except Exception:
            pass

    temp_dir = session.get("temp_dir")
    if not temp_dir or not Path(temp_dir).exists():
        temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
        num_frames = int(frames.shape[0])
        for i in range(num_frames):
            array = frames[i].detach().cpu().numpy()
            array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(array).save(os.path.join(temp_dir, f"{i:05d}.jpg"))
        session["temp_dir"] = temp_dir

    with torch.inference_mode():
        with _video_autocast(video_model):
            response = video_model.start_session(resource_path=temp_dir, session_id=None)
    session["session_id"] = response["session_id"]

    old_cache = _SESSION_FRAMES_CACHE.pop(old_session_id, None)
    if old_cache:
        _SESSION_FRAMES_CACHE[session["session_id"]] = old_cache

    for entry in session.get("_prompts_seq", []):
        if skip_node is not None and str(entry.get("node")) == str(skip_node):
            continue
        model = getattr(video_model, "model", None)
        old_threshold = getattr(model, "score_threshold_detection", None)
        try:
            if old_threshold is not None:
                model.score_threshold_detection = float(entry.get("threshold", old_threshold))
            _video_add_prompt(
                session=session,
                frame_idx=entry["frame_idx"],
                obj_id=entry["obj_id"],
                points=entry.get("points"),
                point_labels=entry.get("point_labels"),
                boxes=entry.get("boxes"),
                box_labels=entry.get("box_labels"),
            )
        finally:
            if old_threshold is not None:
                model.score_threshold_detection = old_threshold


def _ensure_video_model_on_device(video_model):
    model = getattr(video_model, "model", None)
    if model is None:
        return False
    try:
        current_device = next(model.parameters()).device
    except StopIteration:
        return False
    target_device = (torch.device("cuda", torch.cuda.current_device())
                     if torch.cuda.is_available() else torch.device("cpu"))
    if current_device != target_device:
        model.to(device=target_device)
        return True
    return False


def _video_autocast(video_model):
    model = getattr(video_model, "model", None)
    if model is None:
        return nullcontext()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _video_add_prompt(session, frame_idx, obj_id, points=None, point_labels=None,
                     boxes=None, box_labels=None):
    video_model = session["model"]
    _ensure_video_model_on_device(video_model)
    with torch.inference_mode():
        with _video_autocast(video_model):
            return video_model.add_prompt(
                session_id=session["session_id"],
                frame_idx=int(frame_idx),
                text=None,
                points=points,
                point_labels=point_labels,
                bounding_boxes=boxes,
                bounding_box_labels=box_labels,
                obj_id=int(obj_id),
            )


def _is_recoverable_session_error(error):
    message = str(error).lower()
    markers = (
        "cannot find session", "might have expired",
        "input type", "weight type", "bias type", "should be the same",
        "expected all tensors to be on the same device",
    )
    return any(m in message for m in markers)


# =============================================================================
# REST Endpoints for Pre-Queue Frame Preview & Live Editing
# =============================================================================
@PromptServer.instance.routes.post("/sam3/prepare_frames")
async def sam3_prepare_frames(request):
    try:
        body = await request.json()
        key = body.get("preview_key")
        filename = body.get("filename") or body.get("video") or body.get("image")

        if not filename and "node" in body:
            inputs = body["node"].get("inputs", {})
            for k in ("video", "image", "file", "filename", "upload"):
                if k in inputs and isinstance(inputs[k], str):
                    filename = inputs[k]
                    break

        if not key or not filename:
            return web.json_response({"error": "missing preview_key or filename"}, status=400)

        cached = _PREVIEW_CACHE.get(key)
        if cached is not None:
            return web.json_response({"cached": True, "n": cached["n"],
                                      "w": cached["w"], "h": cached["h"],
                                      "preview_key": key})

        loop = asyncio.get_event_loop()
        b64_list, w, h = await loop.run_in_executor(None, _extract_frames_from_file, filename)
        n = len(b64_list)
        _PREVIEW_CACHE[key] = {"b64": b64_list, "w": w, "h": h, "n": n}
        return web.json_response({"cached": False, "n": n, "w": w, "h": h, "preview_key": key})
    except Exception as e:
        log.exception("prepare_frames failed")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/sam3/preview_frames/{key}")
async def sam3_preview_meta(request):
    key = request.match_info["key"]
    info = _PREVIEW_CACHE.get(key)
    if info is None:
        return web.json_response({"error": "not cached"}, status=404)
    return web.json_response({"preview_key": key, "n": info["n"],
                              "w": info["w"], "h": info["h"]})


@PromptServer.instance.routes.get("/sam3/preview_frames/{key}/{idx}")
async def sam3_preview_frame(request):
    key = request.match_info["key"]
    try:
        idx = int(request.match_info["idx"])
    except Exception:
        return web.json_response({"error": "bad idx"}, status=400)
    info = _PREVIEW_CACHE.get(key)
    if info is None:
        return web.json_response({"error": "not cached"}, status=404)
    if idx < 0 or idx >= len(info["b64"]):
        return web.json_response({"error": "idx out of range"}, status=400)
    return web.json_response({"preview_key": key, "idx": idx,
                              "n": info["n"], "w": info["w"], "h": info["h"],
                              "b64": info["b64"][idx]})


@PromptServer.instance.routes.get("/sam3/video_frames/list")
async def sam3_video_frames_list(request):
    return web.json_response({"sessions": [
        {"sid": sid, "n": v["n"], "w": v["w"], "h": v["h"]}
        for sid, v in _SESSION_FRAMES_CACHE.items()
    ]})


@PromptServer.instance.routes.get("/sam3/video_frames/{sid}")
async def sam3_video_frames_meta(request):
    sid = request.match_info["sid"]
    info = _SESSION_FRAMES_CACHE.get(sid)
    if info is None:
        return web.json_response({"error": "not cached"}, status=404)
    return web.json_response({"sid": sid, "n": info["n"],
                              "w": info["w"], "h": info["h"]})


@PromptServer.instance.routes.get("/sam3/video_frames/{sid}/{idx}")
async def sam3_video_frames_get(request):
    sid = request.match_info["sid"]
    try:
        idx = int(request.match_info["idx"])
    except Exception:
        return web.json_response({"error": "bad idx"}, status=400)
    info = _SESSION_FRAMES_CACHE.get(sid)
    if info is None:
        return web.json_response({"error": "not cached"}, status=404)
    if idx < 0 or idx >= len(info["b64"]):
        return web.json_response({"error": "idx out of range"}, status=400)
    return web.json_response({"sid": sid, "idx": idx, "n": info["n"],
                              "w": info["w"], "h": info["h"],
                              "b64": info["b64"][idx]})


# =============================================================================
# Nodes
# =============================================================================
class SAM3VideoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {"default": "", "multiline": False}),
                "use_gpu_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {"hf_token": ("STRING", {"default": ""})}
        }

    RETURN_TYPES = ("SAM3_VIDEO_MODEL",)
    RETURN_NAMES = ("video_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3/video"

    def load_model(self, checkpoint_path="", use_gpu_cache=True, hf_token=""):
        resolved = self._resolve_checkpoint(checkpoint_path)
        global _VIDEO_MODEL_CACHE
        if resolved in _VIDEO_MODEL_CACHE:
            predictor = _VIDEO_MODEL_CACHE[resolved]
            predictor.use_gpu_cache = use_gpu_cache
            if use_gpu_cache and hasattr(predictor, "model") and torch.cuda.is_available():
                predictor.model.to("cuda")
            return (predictor,)

        bpe = Path(__file__).parent / "sam3_lib" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe.exists():
            bpe = Path(__file__).parent.parent / "sam3" / "bpe_simple_vocab_16e6.txt.gz"

        predictor = build_sam3_video_predictor(
            checkpoint_path=resolved, bpe_path=str(bpe),
            hf_token=hf_token if hf_token else None, gpus_to_use=None,
        )
        predictor.use_gpu_cache = use_gpu_cache
        predictor.model.eval()
        _VIDEO_MODEL_CACHE[resolved] = predictor
        return (predictor,)

    @staticmethod
    def _resolve_checkpoint(user_path):
        from folder_paths import base_path as comfy_base
        if user_path and user_path.strip():
            p = Path(user_path.strip())
            if p.exists() and p.is_file():
                return str(p.resolve())
            raise FileNotFoundError(f"[SAM3 Video] Not found: {user_path}")
        models_dir = Path(comfy_base) / "models" / "sam3"
        for name in ("sam3.safetensors", "sam3.pt"):
            c = models_dir / name
            if c.exists() and c.is_file() and c.stat().st_size > 1_000_000:
                return str(c.resolve())
        raise FileNotFoundError(f"[SAM3 Video] No checkpoint found in {models_dir}")


class SAM3InitVideoSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_model": ("SAM3_VIDEO_MODEL",),
                "video_frames": ("IMAGE", {"tooltip": "Video frames batch [N, H, W, C]."}),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),
                "score_threshold_detection": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "new_det_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, video_frames=None, session_id="", **kwargs):
        params = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if k != "video_model")
        return f"{session_id}|{_frames_signature(video_frames)}|{params}"

    RETURN_TYPES = ("SAM3_VIDEO_SESSION", "STRING")
    RETURN_NAMES = ("session", "session_id")
    FUNCTION = "init_session"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True

    def init_session(self, video_model, video_frames, session_id="",
                     score_threshold_detection=0.3, new_det_thresh=0.4):
        m = video_model.model
        m.score_threshold_detection = score_threshold_detection
        m.new_det_thresh = new_det_thresh
        m.fill_hole_area = 16
        m.assoc_iou_thresh = 0.1
        m.det_nms_thresh = 0.1
        m.hotstart_unmatch_thresh = 8
        m.hotstart_dup_thresh = 8
        m.init_trk_keep_alive = 30
        m.hotstart_delay = 15
        m.decrease_trk_keep_alive_for_empty_masklets = False
        m.suppress_unmatched_only_within_hotstart = True

        sid_to_use = session_id if session_id else None
        if sid_to_use:
            try:
                video_model.get_session_state(sid_to_use)
            except Exception:
                import uuid
                sid_to_use = f"auto_recovered_{uuid.uuid4().hex[:8]}"

        temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
        num_frames = int(video_frames.shape[0])
        for i in range(num_frames):
            frame = (video_frames[i].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).save(os.path.join(temp_dir, f"{i:05d}.jpg"))

        try:
            response = video_model.start_session(resource_path=temp_dir, session_id=sid_to_use)
        except Exception:
            import uuid
            sid_to_use = f"emergency_{uuid.uuid4().hex[:8]}"
            response = video_model.start_session(resource_path=temp_dir, session_id=sid_to_use)

        actual_session_id = response["session_id"]
        h = int(video_frames.shape[1])
        w = int(video_frames.shape[2])

        session_data = {
            "model": video_model,
            "session_id": actual_session_id,
            "temp_dir": temp_dir,
            "num_frames": num_frames,
            "height": h, "width": w,
            "frames": video_frames.detach().cpu(),
            "_prompt_history": [],
        }

        b64_frames = _encode_frames_for_ui(video_frames)
        _SESSION_FRAMES_CACHE[actual_session_id] = {
            "b64": b64_frames, "w": w, "h": h, "n": num_frames,
        }
        session_data["_ui_frames_b64"] = b64_frames
        session_data["_ui_cache_key"] = f"sam3-video:{actual_session_id}"

        return {
            "ui": {
                "session_id": [actual_session_id],
                "num_frames": [num_frames],
                "width": [w], "height": [h],
            },
            "result": (session_data, actual_session_id),
        }


class SAM3InitVideoSessionAdvanced(SAM3InitVideoSession):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"].update({
            "fill_hole_area": ("INT", {"default": 16, "min": 0, "max": 1000}),
            "assoc_iou_thresh": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            "det_nms_thresh": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            "hotstart_unmatch_thresh": ("INT", {"default": 8, "min": 0, "max": 999}),
            "hotstart_dup_thresh": ("INT", {"default": 8, "min": 0, "max": 999}),
            "init_trk_keep_alive": ("INT", {"default": 30, "min": -10, "max": 50}),
            "hotstart_delay": ("INT", {"default": 15, "min": 0, "max": 200}),
            "decrease_keep_alive_empty": ("BOOLEAN", {"default": False}),
            "suppress_unmatched_globally": ("BOOLEAN", {"default": True}),
        })
        return base

    def init_session(self, video_model, video_frames, session_id="",
                     score_threshold_detection=0.3, new_det_thresh=0.4,
                     fill_hole_area=16, assoc_iou_thresh=0.1, det_nms_thresh=0.1,
                     hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
                     init_trk_keep_alive=30, hotstart_delay=15,
                     decrease_keep_alive_empty=False, suppress_unmatched_globally=True):
        m = video_model.model
        m.score_threshold_detection = score_threshold_detection
        m.new_det_thresh = new_det_thresh
        m.fill_hole_area = fill_hole_area
        m.assoc_iou_thresh = assoc_iou_thresh
        m.det_nms_thresh = det_nms_thresh
        m.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        m.hotstart_dup_thresh = hotstart_dup_thresh
        m.init_trk_keep_alive = init_trk_keep_alive
        m.hotstart_delay = hotstart_delay
        m.decrease_trk_keep_alive_for_empty_masklets = decrease_keep_alive_empty
        m.suppress_unmatched_only_within_hotstart = not suppress_unmatched_globally
        return super().init_session(video_model, video_frames, session_id,
                                    score_threshold_detection, new_det_thresh)


class SAM3VideoPromptEditor:
    DESCRIPTION = """
### SAM3 Video Prompt Editor

One node = one frame = one **obj_id**. Chain several editors for multi-frame / multi-object prompts.

#### Controls
- **Ctrl+LMB** — positive point / start **positive** box  
- **Ctrl+RMB** — negative point / start **negative** box  
- **LMB drag** — move point or box (handles = resize)  
- **RMB** — delete under cursor  
- **MMB drag** — pan image  
- **Wheel** — Comfy graph zoom (overlay follows)  
- **F** / double-click — fit  
- **Slider** / ◀ ▶ / arrow keys — scrub frames  

Mode **points** / **boxes**: only one type is active; switching clears the other.

Badges show **index within type** (4 counters): +points, −points, +boxes, −boxes (each 0,1,2…).

---

#### How boxes work in SAM 3 (PCS)

SAM 3 uses **Promptable Concept Segmentation**: visual boxes act as **exemplars**.

- **Positive box (label 1)** — “segment everything like this.”  
- **Negative box (label 0)** — “ignore this and things like it” (e.g. referee vs players).

On **images**, the grounding/image path fully supports positive + negative box labels.

On **video tracker / propagate**, the session tracker often treats a box mainly as **object init** (SAM2-style). For rejecting false tracks on video, **negative points** (and/or concept/text prompts) are usually more reliable than relying on negative boxes alone during propagation. Positive boxes still work well to seed an object; negative boxes are stored and sent when the API accepts `bounding_box_labels`, but results can vary by SAM3 video build.

---

#### Workflow
`Load Video` → `Init Video Session` → **this Editor** → `Propagate` → `Output`
"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"session": ("SAM3_VIDEO_SESSION",)},
            "optional": {
                "prompt_mode": (["points", "boxes"], {"default": "points"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "obj_id": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "score_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "points_json": ("STRING", {"default": "[]", "multiline": True}),
                "boxes_json": ("STRING", {"default": "[]", "multiline": True}),
                "ui_cache_key": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SAM3_VIDEO_SESSION",)
    RETURN_NAMES = ("session",)
    FUNCTION = "apply"
    CATEGORY = "SAM3/video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, prompt_mode="points", frame_index=0, obj_id=1,
                   score_threshold=0.3, points_json="[]", boxes_json="[]", **kw):
        return f"{prompt_mode}|{frame_index}|{obj_id}|{score_threshold}|{points_json}|{boxes_json}"

    @staticmethod
    def _load(s, fb):
        try:
            return json.loads(s) if s else fb
        except Exception:
            return fb

    def apply(self, session, prompt_mode="points", frame_index=0, obj_id=1,
              score_threshold=0.30, points_json="[]", boxes_json="[]",
              ui_cache_key="", unique_id="0"):
        if session is None:
            raise ValueError("[SAM3] Session is empty")
        vm = session["model"]
        _ensure_video_model_on_device(vm)

        frames = session.get("frames")
        if frames is None:
            raise ValueError("[SAM3] No frames in session. Ensure Init Session ran.")

        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        frame_index = max(0, min(int(frame_index), n - 1))
        obj_id = int(obj_id)
        score_threshold = float(score_threshold)

        ck = "_ui_frames_b64"
        if ck in session:
            all_b64 = session[ck]
        else:
            all_b64 = _encode_frames_for_ui(frames)
            session[ck] = all_b64
            _SESSION_FRAMES_CACHE[session["session_id"]] = {
                "b64": all_b64, "w": w, "h": h, "n": n,
            }

        pts = self._load(points_json, [])
        boxes = self._load(boxes_json, [])
        cache_key = session.get("_ui_cache_key") or f"sam3-video:{session['session_id']}"
        session["_ui_cache_key"] = cache_key

        point_coords = point_labels = api_boxes = api_box_labels = None
        if prompt_mode == "points" and pts:
            point_coords = [[float(p["x"]), float(p["y"])] for p in pts]
            point_labels = [int(p.get("label", 1)) for p in pts]
        elif prompt_mode == "boxes" and boxes:
            api_boxes, api_box_labels = [], []
            for b in boxes:
                x0, y0 = min(b["x0"], b["x1"]), min(b["y0"], b["y1"])
                x1, y1 = max(b["x0"], b["x1"]), max(b["y0"], b["y1"])
                api_boxes.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
                api_box_labels.append(1 if b.get("positive", True) else 0)

        has_prompt = bool(point_coords or api_boxes)
        entry = {
            "node": unique_id, "frame_idx": frame_index, "obj_id": obj_id,
            "threshold": score_threshold,
            "points": point_coords, "point_labels": point_labels,
            "boxes": api_boxes, "box_labels": api_box_labels,
            "cache_key": [cache_key],
        }
        seq = session.setdefault("_prompts_seq", [])
        for i, e in enumerate(seq):
            if e.get("node") == unique_id:
                if has_prompt:
                    seq[i] = entry
                else:
                    seq.pop(i)
                break
        else:
            if has_prompt:
                seq.append(entry)

        was_dead = not _session_alive(vm, session.get("session_id"))
        if was_dead:
            _recover_session(session)
            vm = session["model"]
            _ensure_video_model_on_device(vm)

        committed = False
        sig = (f"{unique_id}|{frame_index}|{obj_id}|{prompt_mode}|"
               f"{score_threshold}|{points_json}|{boxes_json}")
        done = session.setdefault("_committed", {})

        if was_dead:
            if has_prompt:
                done[unique_id] = sig
                committed = True
        elif has_prompt and done.get(unique_id) != sig:
            m = getattr(vm, "model", None)
            old = getattr(m, "score_threshold_detection", None)
            try:
                if old is not None:
                    m.score_threshold_detection = score_threshold
                try:
                    _video_add_prompt(
                        session=session, frame_idx=frame_index, obj_id=obj_id,
                        points=point_coords, point_labels=point_labels,
                        boxes=api_boxes, box_labels=api_box_labels,
                    )
                except RuntimeError as e:
                    if not _is_recoverable_session_error(e):
                        raise
                    _recover_session(session, skip_node=unique_id)
                    _video_add_prompt(
                        session=session, frame_idx=frame_index, obj_id=obj_id,
                        points=point_coords, point_labels=point_labels,
                        boxes=api_boxes, box_labels=api_box_labels,
                    )
            finally:
                if old is not None:
                    m.score_threshold_detection = old
            done[unique_id] = sig
            session.setdefault("_prompt_history", []).append({
                "node": unique_id, "frame_idx": frame_index, "obj_id": obj_id,
                "mode": prompt_mode, "threshold": score_threshold,
            })
            committed = True

        return {
            "ui": {
                "all_frames": all_b64, "num_frames": [n], "width": [w], "height": [h],
                "frame_index": [frame_index], "cache_key": [cache_key], "committed": [committed],
                "session_id": [session["session_id"]],
            },
            "result": (session,),
        }


class SAM3PropagateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"session": ("SAM3_VIDEO_SESSION",)},
            "optional": {
                "propagation_direction": (["both", "forward", "backward"], {"default": "both"}),
                "start_frame_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "max_frames": ("INT", {"default": -1, "min": -1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("SAM3_VIDEO_MASKS", "SAM3_VIDEO_SESSION")
    RETURN_NAMES = ("video_masks", "session")
    FUNCTION = "propagate"
    CATEGORY = "SAM3/video"

    def propagate(self, session, propagation_direction="both",
                  start_frame_index=0, max_frames=-1):
        video_model = session["model"]
        moved = _ensure_video_model_on_device(video_model)
        if moved or not _session_alive(video_model, session.get("session_id")):
            _recover_session(session)
            video_model = session["model"]

        session_id = session["session_id"]
        num_frames = session["num_frames"]
        max_track = max_frames if max_frames > 0 else num_frames

        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": int(start_frame_index),
            "max_frame_num_to_track": int(max_track),
        }
        all_masks = {}
        all_obj_ids = None
        with torch.inference_mode():
            with _video_autocast(video_model):
                for response in video_model.handle_stream_request(request):
                    frame_idx = response["frame_index"]
                    outputs = response["outputs"]
                    all_masks[frame_idx] = outputs
                    if all_obj_ids is None:
                        all_obj_ids = outputs.get("obj_ids", [])
        if all_obj_ids is None:
            all_obj_ids = []
        return ({"session": session, "masks": all_masks,
                 "obj_ids": all_obj_ids, "num_frames": num_frames}, session)


class SAM3VideoOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video_masks": ("SAM3_VIDEO_MASKS",)},
            "optional": {
                "obj_id_filter": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "output_masks"
    CATEGORY = "SAM3/video"

    def output_masks(self, video_masks, obj_id_filter=-1):
        masks_dict = video_masks["masks"]
        num_frames = video_masks["num_frames"]
        session = video_masks["session"]
        height, width = session["height"], session["width"]
        output = torch.zeros((num_frames, height, width), dtype=torch.float32)

        for frame_idx in range(num_frames):
            if frame_idx not in masks_dict:
                continue
            fo = masks_dict[frame_idx]
            if "video_res_masks" in fo:
                fm = fo["video_res_masks"]
            elif "pred_masks" in fo:
                fm = fo["pred_masks"]
                if fm.shape[-2:] != (height, width):
                    fm = torch.nn.functional.interpolate(
                        fm.float(), size=(height, width),
                        mode="bilinear", align_corners=False)
            elif "out_binary_masks" in fo:
                fm = torch.from_numpy(fo["out_binary_masks"])
                if fm.ndim == 3:
                    fm = fm.unsqueeze(1)
                if fm.shape[-2:] != (height, width):
                    fm = torch.nn.functional.interpolate(
                        fm.float(), size=(height, width),
                        mode="bilinear", align_corners=False) > 0.5
            else:
                continue

            ids = fo.get("obj_ids", [])
            if obj_id_filter > 0:
                try:
                    if hasattr(ids, "tolist"): ids = ids.tolist()
                    j = ids.index(obj_id_filter)
                    mask = fm[j, 0] > 0.0
                except (ValueError, IndexError):
                    mask = torch.zeros((height, width), dtype=torch.bool)
            else:
                mask = (fm[:, 0] > 0.0).any(dim=0)
            output[frame_idx] = mask.float().cpu()
        return (output,)


class SAM3CloseVideoSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("SAM3_VIDEO_SESSION",),
                "close_session": ("BOOLEAN", {"default": False,
                    "label_on": "Close session", "label_off": "Keep session alive"}),
                "cleanup_temp_files": ("BOOLEAN", {"default": True,
                    "label_on": "Delete temp files", "label_off": "Keep temp files"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "close_session"
    CATEGORY = "SAM3/video"

    def close_session(self, session, close_session=False, cleanup_temp_files=True):
        video_model = session["model"]
        sid = session["session_id"]
        tmp_dir = session.get("temp_dir")
        if not close_session:
            return (f"Session {sid} kept alive.",)
        try:
            video_model.close_session(sid)
            status = f"Session {sid} closed"
        except Exception as e:
            status = f"Close warning: {e}"
        _SESSION_FRAMES_CACHE.pop(sid, None)
        if cleanup_temp_files and tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            status += "; temp cleaned"
        if (not getattr(video_model, "use_gpu_cache", True)
                and hasattr(video_model, "model")):
            video_model.model.to("cpu")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            import gc; gc.collect()
        return (status,)


NODE_CLASS_MAPPINGS = {
    "SAM3VideoModelLoader": SAM3VideoModelLoader,
    "SAM3InitVideoSession": SAM3InitVideoSession,
    "SAM3InitVideoSessionAdvanced": SAM3InitVideoSessionAdvanced,
    "SAM3VideoPromptEditor": SAM3VideoPromptEditor,
    "SAM3PropagateVideo": SAM3PropagateVideo,
    "SAM3VideoOutput": SAM3VideoOutput,
    "SAM3CloseVideoSession": SAM3CloseVideoSession,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoModelLoader": "SAM3 Load Video Model",
    "SAM3InitVideoSession": "SAM3 Init Video Session",
    "SAM3InitVideoSessionAdvanced": "SAM3 Init Video Session (Advanced)",
    "SAM3VideoPromptEditor": "SAM3 Video Prompt Editor",
    "SAM3PropagateVideo": "SAM3 Propagate Video",
    "SAM3VideoOutput": "SAM3 Video Output",
    "SAM3CloseVideoSession": "SAM3 Close Video Session",
}