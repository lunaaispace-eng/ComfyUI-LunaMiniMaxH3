// Luna collapse — a chevron on the title bar that folds a node's settings away,
// leaving the title, the sockets and anything the node draws for itself.
//
// Written for the two save nodes, whose widget stacks are the tallest thing on
// the canvas (Save Video Simple is 20 widgets and ~660px). Collapsed, it keeps
// its video player, the first/last-frame checkboxes, Autoplay and Download,
// because those live inside a DOM widget rather than in the widget stack.
//
// The rule is exactly that: **canvas widgets fold, DOM widgets stay.** It is not
// a special case per node — it is what "settings" versus "the thing itself"
// means on a ComfyUI node, and it gives Save Image (Simple) the right answer for
// free, since its preview is ComfyUI's own image area and not a widget at all.
//
// Same title-bar mechanics as luna_help.mjs — draw in onDrawForeground at
// negative y, hit-test node-relative pos. A single click is safe up there; the
// rename editor is bound to DOUBLE-click, which onDblClick below swallows.

import { LUNA } from "./luna_theme.mjs";

const ICON_R = 7;
const ICON_INSET = 36;      // the ⓘ owns size[0] - 16, so sit one slot left of it
const PROP = "lunaSettingsCollapsed";

function iconCentre(node) {
    const LG = globalThis.LiteGraph ?? globalThis.comfyAPI?.litegraph?.LiteGraph;
    const h = LG?.NODE_TITLE_HEIGHT ?? 30;
    return [node.size[0] - ICON_INSET, -h / 2];
}

const hit = (node, pos) => {
    const [cx, cy] = iconCentre(node);
    return Math.hypot(pos[0] - cx, pos[1] - cy) <= ICON_R + 3;
};

// Widgets with an .element are DOM widgets — the player, the cards, the preview.
// Those are the node's content, not its settings, so they never fold.
const foldable = (node) => (node.widgets ?? []).filter((w) => !w.element);

/**
 * Fold or unfold a node's settings.
 * `resize: false` when restoring a saved workflow — the stored size already
 * reflects the collapsed state, so recomputing it would double-apply.
 */
export function setCollapsed(node, on, { resize = true } = {}) {
    const targets = foldable(node);
    if (!targets.length) return;

    const before = node.computeSize()[1];
    for (const w of targets) {
        if (on) {
            // Remember what each widget was, so unfolding cannot un-hide something
            // the node itself had hidden. Save Video Simple hides log_level,
            // save_first_frame and save_last_frame via draw/computeSize overrides
            // and re-draws them inside its own panel — those must stay hidden.
            if (w._lunaWasHidden === undefined) w._lunaWasHidden = !!w.hidden;
            w.hidden = true;
        } else if (w._lunaWasHidden !== undefined) {
            w.hidden = w._lunaWasHidden;
            delete w._lunaWasHidden;
        }
    }

    node.properties = node.properties || {};
    node.properties[PROP] = !!on;

    if (resize) {
        // Shrink by exactly the widget height that disappeared, rather than
        // snapping to the computed minimum. Save Image (Simple) keeps whatever
        // extra room it was holding for its preview image, which is not a widget
        // and so contributes nothing to computeSize.
        const after = node.computeSize()[1];
        node.setSize([node.size[0], Math.max(node.size[1] - (before - after), after)]);
    }
    node.setDirtyCanvas(true, true);
}

function drawIcon(node, ctx) {
    const [cx, cy] = iconCentre(node);
    const on = !!node.properties?.[PROP];
    const live = node._lunaCollapseHover;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, ICON_R, 0, Math.PI * 2);
    ctx.strokeStyle = live ? LUNA.accent : LUNA.muted;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    // Chevron points the way the node will move: up folds away, down unfolds.
    ctx.beginPath();
    ctx.lineWidth = 1.6;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const d = on ? 1 : -1;
    ctx.moveTo(cx - 3.2, cy + d * 1.6);
    ctx.lineTo(cx, cy - d * 1.6);
    ctx.lineTo(cx + 3.2, cy + d * 1.6);
    ctx.stroke();
    ctx.restore();
}

/**
 * Add the collapse chevron to every node whose class name passes `match`.
 *   registerLunaCollapse(app, ["SaveImageSimple"], "MyPack.Collapse")
 */
export function registerLunaCollapse(app, match, extensionName) {
    const wanted = typeof match === "function" ? match : (n) => match.includes(n);
    app.registerExtension({
        name: extensionName,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (!wanted(nodeData?.name)) return;

            const onDraw = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDraw?.apply(this, arguments);
                if (!this.flags?.collapsed && ctx) drawIcon(this, ctx);
                return r;
            };

            const onDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function (e, pos) {
                if (!this.flags?.collapsed && hit(this, pos)) {
                    setCollapsed(this, !this.properties?.[PROP]);
                    return true;                       // consumed: no node drag
                }
                return onDown?.apply(this, arguments);
            };

            // The title bar is where LiteGraph opens the rename editor on
            // double-click. A double-press on the chevron must not rename.
            const onDbl = nodeType.prototype.onDblClick;
            nodeType.prototype.onDblClick = function (e, pos) {
                if (!this.flags?.collapsed && hit(this, pos)) return true;
                return onDbl?.apply(this, arguments);
            };

            const onMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function (e, pos) {
                const over = !this.flags?.collapsed && hit(this, pos);
                if (over !== this._lunaCollapseHover) {
                    this._lunaCollapseHover = over;
                    this.setDirtyCanvas(true, false);
                }
                return onMove?.apply(this, arguments);
            };

            // Restore from the workflow.
            //
            // Two frames, then the saved height is re-asserted, because Save Video
            // Simple resizes itself on configure (fitPreviewHeight → setSize of
            // computeSize) and that ran while the widgets were still visible — the
            // node came back folded but 657px tall, all of it empty. The height in
            // the workflow is the truth; nothing here is measured from the layout.
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure?.apply(this, arguments);
                if (this.properties?.[PROP]) {
                    const s = info?.size;
                    const saved = s ? [s[0] ?? s["0"], s[1] ?? s["1"]] : null;
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        setCollapsed(this, true, { resize: false });
                        if (saved && Number.isFinite(saved[1])) this.setSize([saved[0], saved[1]]);
                        this.setDirtyCanvas(true, true);
                    }));
                }
                return r;
            };
        },
    });
}
