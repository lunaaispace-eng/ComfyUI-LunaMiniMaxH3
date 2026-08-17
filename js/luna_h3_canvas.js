// Luna MiniMax H3 Canvas — live readout on the node.
//
// The point of this node is that the numbers are DERIVED, so showing them only
// after a run defeats it. This draws the resolved canvas, frame count and real
// duration as you turn the dials, and greys out `megapixels` while the H3 canvas
// mode is on, since it does nothing there.
//
// The maths below mirrors h3_canvas.py. That file is the source of truth — if the
// model's constants ever change, change them there first, then here.

import { app } from "../../scripts/app.js";
import { LUNA } from "./luna_theme.mjs";

const CANVAS_MULTIPLE = 32;
const BASE_SHORT_EDGE = 768;
const MAX_PIXELS = 768 * 1344;
const FPS = 24;
const FRAME_GRID = 17;
const FRAME_PHASE = 5;
const TRAINED_MIN = 124;
const TRAINED_MAX = 362;

// Landscape form only; the `portrait` widget flips it. See h3_canvas.py.
const ASPECT_RATIOS = {
    "2.39:1 (anamorphic)": [239, 100],
    "21:9 (ultrawide)": [21, 9],
    "2:1": [2, 1],
    "1.91:1 (link preview)": [191, 100],
    "16:9 (widescreen)": [16, 9],
    "16:10": [16, 10],
    "3:2 (photo)": [3, 2],
    "4:3": [4, 3],
    "5:4": [5, 4],
    "1:1 (square)": [1, 1],
};

const roundTo = (v) =>
    Math.max(CANVAS_MULTIPLE, Math.round(v / CANVAS_MULTIPLE) * CANVAS_MULTIPLE);

// JS % keeps the sign of the dividend, Python's does not — hence the double mod.
// Without it every frame count below the phase lands wrong.
const alignFrames = (n) => {
    n = Math.max(FRAME_PHASE, Math.round(n));
    return n + ((((FRAME_PHASE - n) % FRAME_GRID) + FRAME_GRID) % FRAME_GRID);
};

function h3Canvas(ratio) {
    let short = BASE_SHORT_EDGE;
    let long = BASE_SHORT_EDGE * Math.max(ratio, 1 / ratio);
    if (short * long > MAX_PIXELS) {
        const shrink = Math.sqrt(MAX_PIXELS / (short * long));
        short *= shrink;
        long *= shrink;
    }
    const [w, h] = ratio >= 1 ? [long, short] : [short, long];
    return [roundTo(w), roundTo(h)];
}

function megapixelCanvas(ratio, megapixels) {
    const area = Math.max(0.01, megapixels) * 1e6;
    return [roundTo(Math.sqrt(area * ratio)), roundTo(Math.sqrt(area / ratio))];
}

const widgetValue = (node, name) => node.widgets?.find((w) => w.name === name)?.value;

function resolve(node) {
    const base = ASPECT_RATIOS[widgetValue(node, "ratio")] || [16, 9];
    const pair = widgetValue(node, "portrait") ? [base[1], base[0]] : base;
    const ratio = pair[0] / pair[1];
    const mode = widgetValue(node, "size_mode");
    const [width, height] = mode === "megapixels"
        ? megapixelCanvas(ratio, Number(widgetValue(node, "megapixels")) || 1.03)
        : h3Canvas(ratio);

    const asked = Number(widgetValue(node, "duration_seconds")) || 0;
    const length = alignFrames(asked * FPS);
    const enabled = widgetValue(node, "interpolation_enabled") !== false;
    const factor = enabled
        ? Math.max(1, Number(widgetValue(node, "interpolation_factor")) || 1)
        : 1;
    const seconds = length / FPS;

    const notes = [];
    // Half a grid step — see h3_canvas.py. Below that the snap is unavoidable and
    // the main line already shows the real duration.
    if (Math.abs(seconds - asked) >= FRAME_GRID / FPS / 2) notes.push(`asked ${asked}s`);
    if (length < TRAINED_MIN || length > TRAINED_MAX) notes.push("outside trained range");
    if (width * height > MAX_PIXELS) {
        notes.push(`${(width * height / MAX_PIXELS).toFixed(1)}x default canvas`);
    }

    return { width, height, length, seconds, outputFps: FPS * factor, factor, notes, pair };
}

// Hide the widgets that currently do nothing: `megapixels` in H3 canvas mode, and
// `interpolation_factor` while interpolation is gated off. widget.hidden is the flag
// that actually works — computeSize skips hidden widgets, so the node shrinks too.
function syncWidgets(node) {
    const wanted = {
        megapixels: widgetValue(node, "size_mode") !== "megapixels",
        interpolation_factor: widgetValue(node, "interpolation_enabled") === false,
    };
    let changed = false;
    for (const [name, hidden] of Object.entries(wanted)) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (!widget || widget.hidden === hidden) continue;
        widget.hidden = hidden;
        changed = true;
    }
    if (!changed) return;
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

function drawReadout(ctx, node, width, y) {
    const r = resolve(node);
    const pad = 10;
    const x = 14;
    // node.size[0] is authoritative. The width LiteGraph hands a custom widget is
    // not always the node's, and trusting it drew the panel off the side of the node.
    const outer = Number(node?.size?.[0]) || Number(width) || 240;
    const w = Math.max(120, outer - x * 2);
    const h = r.notes.length ? 74 : 56;

    // save() must be paired with restore() even if something below throws, or every
    // node drawn after this one inherits our fill style, font and path.
    ctx.save();
    try {
    ctx.beginPath();
    // roundRect is not on every canvas implementation; a plain rect is a fine panel.
    if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, w, h, 6);
    else ctx.rect(x, y, w, h);
    ctx.fillStyle = LUNA.panel;
    ctx.fill();
    ctx.strokeStyle = LUNA.border;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillStyle = LUNA.accent;
    ctx.font = "600 16px sans-serif";
    ctx.fillText(`${r.width} × ${r.height}`, x + pad, y + 24);

    ctx.textAlign = "right";
    ctx.fillStyle = LUNA.muted;
    ctx.font = "11px sans-serif";
    ctx.fillText(
        `${r.pair[0]}:${r.pair[1]}   ${(r.width * r.height / 1e6).toFixed(2)} MP`,
        x + w - pad, y + 24,
    );

    ctx.textAlign = "left";
    ctx.fillStyle = LUNA.text;
    ctx.font = "11px sans-serif";
    const rate = r.factor > 1 ? `${FPS} → ${r.outputFps} fps` : `${FPS} fps`;
    ctx.fillText(`${r.length} frames  ${r.seconds.toFixed(2)}s  ${rate}`, x + pad, y + 44);

    if (r.notes.length) {
        ctx.fillStyle = LUNA.danger;
        ctx.font = "11px sans-serif";
        ctx.fillText(r.notes.join("   "), x + pad, y + 63);
    }
    } finally {
        ctx.restore();
    }
}

// LiteGraph draws every node in one pass and does not catch. An exception thrown
// from a widget's draw aborts the rest of that frame, so unrelated nodes render
// without their widgets and the graph looks broken. Nothing in this file may throw
// into the host — hence the guards on every entry point.
function guard(label, fn, fallback) {
    try {
        return fn();
    } catch (error) {
        if (!guard.reported.has(label)) {
            guard.reported.add(label);
            console.error(`[Luna H3 Canvas] ${label} failed; readout disabled.`, error);
        }
        return fallback;
    }
}
guard.reported = new Set();

app.registerExtension({
    name: "LunaSaveSimple.H3Canvas",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "LunaMiniMaxH3Canvas") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            const node = this;

            this.addCustomWidget({
                type: "luna_h3_readout",
                name: "readout",
                // Not a real input: never serialise it into the workflow.
                serialize: false,
                value: null,
                options: { serialize: false },
                computeSize() {
                    return [0, guard("computeSize", () => (resolve(node).notes.length ? 82 : 64), 64)];
                },
                draw(ctx, _node, widgetWidth, widgetY) {
                    guard("draw", () => drawReadout(ctx, node, widgetWidth, widgetY), undefined);
                },
            });

            // Relayout whenever a widget feeding the readout changes. Closure over
            // `node` rather than touching `this` — LiteGraph calls widget callbacks
            // as (value, canvas, node, pos, event) with `this` set to the widget,
            // and the original callback needs that left alone.
            for (const widget of this.widgets || []) {
                if (widget.type === "luna_h3_readout") continue;
                const previous = widget.callback;
                widget.callback = function (...args) {
                    const result = previous?.apply(this, args);
                    guard("sync", () => {
                        syncWidgets(node);
                        node.setDirtyCanvas(true, true);
                    }, undefined);
                    return result;
                };
            }

            guard("init", () => {
                syncWidgets(this);
                this.setSize(this.computeSize());
            }, undefined);
        };

        // Loading a saved workflow restores widget values without firing callbacks.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            guard("configure", () => syncWidgets(this), undefined);
        };
    },
});
