import { app } from "../../../scripts/app.js";

const NODE_ID = "SAM3AddVideoPrompt";

// Типы для input-сокетов (кастомные типы, не виджеты)
const SOCKET_TYPES = {
    positive_points: "SAM3_POINTS_PROMPT",
    negative_points: "SAM3_POINTS_PROMPT",
    boxes: "SAM3_BOXES_PROMPT",
};

// text_prompt — обычный STRING виджет, не сокет
const WIDGET_NAMES = ["text_prompt"];

const MODE_SOCKETS = {
    points: { show: ["positive_points", "negative_points"], hide: ["boxes"] },
    text:   { show: [], hide: ["positive_points", "negative_points", "boxes"] },
    boxes:  { show: ["boxes"], hide: ["positive_points", "negative_points"] },
};

const MODE_WIDGETS = {
    points: { show: [], hide: ["text_prompt"] },
    text:   { show: ["text_prompt"], hide: [] },
    boxes:  { show: [], hide: ["text_prompt"] },
};

// ---------- Виджеты (text_prompt) ----------

function setWidgetVisible(node, name, visible) {
    const widget = node.widgets?.find(w => w.name === name);
    if (!widget) return;
    if (!widget._sam3Orig) {
        widget._sam3Orig = { type: widget.type, computeSize: widget.computeSize };
    }
    if (visible) {
        widget.type = widget._sam3Orig.type;
        widget.computeSize = widget._sam3Orig.computeSize;
        widget.hidden = false;
    } else {
        widget.hidden = true;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }
}

// ---------- Сокеты (positive_points/negative_points/boxes) ----------

function findInputIndex(node, name) {
    if (!node.inputs) return -1;
    return node.inputs.findIndex(i => i.name === name);
}

function removeSocket(node, name) {
    const idx = findInputIndex(node, name);
    if (idx === -1) return; // уже скрыт

    // Отключаем линк перед удалением, чтобы не осталось "битых" связей
    if (node.inputs[idx].link != null) {
        node.disconnectInput(idx);
    }
    node.removeInput(idx);
}

function addSocket(node, name) {
    const idx = findInputIndex(node, name);
    if (idx !== -1) return; // уже показан

    const type = SOCKET_TYPES[name];
    node.addInput(name, type);

    // Восстанавливаем tooltip, если он был задан на схеме ноды
    const inputDef = node.constructor?.nodeData?.input?.optional?.[name];
    if (inputDef && node.inputs?.length) {
        const newInput = node.inputs[node.inputs.length - 1];
        if (newInput && inputDef[1]?.tooltip) {
            newInput.tooltip = inputDef[1].tooltip;
        }
    }
}

// ---------- Общая логика ----------

function refreshModeUI(node) {
    const modeWidget = node.widgets?.find(w => w.name === "prompt_mode");
    if (!modeWidget) return;
    const mode = modeWidget.value;

    // Widgets
    const wcfg = MODE_WIDGETS[mode] || { show: [], hide: [] };
    for (const n of wcfg.show) setWidgetVisible(node, n, true);
    for (const n of wcfg.hide) setWidgetVisible(node, n, false);

    // Sockets — сначала прячем ненужные, затем добавляем нужные
    const scfg = MODE_SOCKETS[mode] || { show: [], hide: [] };
    for (const n of scfg.hide) removeSocket(node, n);
    for (const n of scfg.show) addSocket(node, n);

    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "sam3.promptMode",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        // Сохраняем исходную схему для восстановления tooltip при возврате сокета
        nodeType.nodeData = nodeData;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            const modeWidget = node.widgets?.find(w => w.name === "prompt_mode");
            if (modeWidget) {
                const origCb = modeWidget.callback;
                modeWidget.callback = function (...args) {
                    origCb?.apply(this, args);
                    refreshModeUI(node);
                };
            }

            // Для новой ноды (не из сохранённого воркфлоу) применяем сразу
            requestAnimationFrame(() => refreshModeUI(node));
            return r;
        };

        // Для ноды, восстановленной из сохранённого JSON воркфлоу
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            const node = this;
            // Ждём, пока ComfyUI полностью восстановит все связи графа
            requestAnimationFrame(() => {
                requestAnimationFrame(() => refreshModeUI(node));
            });
            return r;
        };
    },
});