// Luna node help — an ⓘ on the title bar that opens the node's own description.
//
// ComfyUI does NOT give you this. A node's DESCRIPTION shows as a tooltip in the
// node library, but nothing appears on the node itself, which is why our nodes
// looked bare next to packs that ship their own. This draws the icon and the
// popup; the text is whatever the Python DESCRIPTION already says, so there is no
// second copy of the documentation to keep in step.
//
// The approach (canvas-drawn icon, hit test, DOM popup, markdown via the
// extension manager) is the same one ComfyUI-Deno uses. Written here from the
// LiteGraph API rather than copied — that pack is GPL-3.0 and this one is
// Apache-2.0.

import { LUNA } from "./luna_theme.mjs";

const ICON_R = 7;          // radius of the ⓘ circle
const ICON_INSET = 16;     // from the node's right edge

/* ------------------------------------------------------------------- popup */

function ensureCSS() {
    if (document.getElementById("luna-help-css")) return;
    const s = document.createElement("style");
    s.id = "luna-help-css";
    s.textContent = `
.luna-help-back { position:fixed; inset:0; z-index:9998; background:rgba(0,0,0,.45); }
.luna-help { position:fixed; z-index:9999; left:50%; top:50%; transform:translate(-50%,-50%);
    width:min(560px, 88vw); max-height:78vh; display:flex; flex-direction:column;
    background:${LUNA.panel}; color:${LUNA.text}; border:1px solid ${LUNA.border};
    border-radius:${LUNA.radius}px; box-shadow:0 18px 50px rgba(0,0,0,.55);
    font-family:"Segoe UI", system-ui, sans-serif; font-size:13px; }
.luna-help-h { display:flex; align-items:center; gap:8px; padding:10px 12px;
    border-bottom:1px solid ${LUNA.border}; }
.luna-help-h b { flex:1 1 auto; font-size:13px; }
.luna-help-x { cursor:pointer; padding:2px 8px; border-radius:6px; color:${LUNA.muted}; }
.luna-help-x:hover { background:${LUNA.danger}; color:#fff; }
.luna-help-b { padding:12px 14px; overflow:auto; line-height:1.5; }
.luna-help-b p { margin:0 0 .7em; }
.luna-help-b code { background:${LUNA.bg}; padding:1px 4px; border-radius:4px; }
.luna-help-b h1,.luna-help-b h2,.luna-help-b h3 { font-size:13px; margin:.9em 0 .4em; color:${LUNA.accent}; }
.luna-help-b table { border-collapse:collapse; width:100%; margin:.4em 0 .9em; }
.luna-help-b td,.luna-help-b th { border:1px solid ${LUNA.border}; padding:3px 6px; text-align:left; }
.luna-help-io { margin-top:.4em; }
.luna-help-io div { display:flex; gap:8px; padding:3px 0; border-top:1px solid ${LUNA.border}; }
.luna-help-io span:first-child { color:${LUNA.accent}; min-width:120px; }
.luna-tip { position:fixed; z-index:10000; display:none; max-width:300px; padding:6px 9px;
    background:${LUNA.panel}; color:${LUNA.text}; border:1px solid ${LUNA.border};
    border-radius:6px; box-shadow:0 6px 20px rgba(0,0,0,.5); pointer-events:none;
    font-family:"Segoe UI", system-ui, sans-serif; font-size:11px; line-height:1.45; }
`;
    document.head.appendChild(s);
}

const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function toHtml(text) {
    const md = globalThis.app?.extensionManager?.renderMarkdownToHtml;
    if (typeof md === "function") {
        try { return md(text); } catch { /* fall through */ }
    }
    return esc(text).split(/\n\s*\n/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function openHelp(node) {
    ensureCSS();
    const def = node.constructor?.nodeData ?? {};
    const body = def.description || node.description || "No description for this node yet.";

    const back = document.createElement("div");
    back.className = "luna-help-back";
    const box = document.createElement("div");
    box.className = "luna-help";
    const close = () => { back.remove(); box.remove(); document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };

    box.innerHTML =
        `<div class="luna-help-h"><b>${esc(def.display_name || node.title)}</b>` +
        `<div class="luna-help-x">✕</div></div><div class="luna-help-b"></div>`;
    const bodyEl = box.querySelector(".luna-help-b");
    bodyEl.innerHTML = toHtml(body);

    // Input tooltips are documentation too, and every Luna node already has them.
    // Listing them here means the popup covers the whole node without anyone
    // writing a separate help file.
    const req = def.input?.required ?? {};
    const opt = def.input?.optional ?? {};
    const rows = [];
    for (const [name, spec] of [...Object.entries(req), ...Object.entries(opt)]) {
        const tip = spec?.[1]?.tooltip;
        if (tip) rows.push(`<div><span>${esc(name)}</span><span>${esc(tip)}</span></div>`);
    }
    if (rows.length) {
        bodyEl.insertAdjacentHTML("beforeend",
            `<h3>Inputs</h3><div class="luna-help-io">${rows.join("")}</div>`);
    }

    // Outputs come from the node's OUTPUT_TOOLTIPS, the same text ComfyUI shows
    // when you hover a socket. Listing them here means one definition serves the
    // hover and the panel, so they can never drift apart.
    const names = def.output_name ?? def.output ?? [];
    const tips = def.output_tooltips ?? [];
    const outRows = names
        .map((nm, i) => (tips[i] ? `<div><span>${esc(nm)}</span><span>${esc(tips[i])}</span></div>` : ""))
        .filter(Boolean);
    if (outRows.length) {
        bodyEl.insertAdjacentHTML("beforeend",
            `<h3>Outputs</h3><div class="luna-help-io">${outRows.join("")}</div>`);
    }

    box.querySelector(".luna-help-x").onclick = close;
    back.onclick = close;
    document.addEventListener("keydown", onKey);
    document.body.append(back, box);
}

/* ------------------------------------------------- output-socket tooltips */

// ComfyUI sends OUTPUT_TOOLTIPS in object_info but its frontend (1.48.x) never
// attaches them to output slots — verified against the core VAEDecode node,
// which has one and shows nothing either. Input/widget tooltips are handled by
// ComfyUI; outputs are not. So this draws them.

let tipEl = null;

function showTip(text, x, y) {
    ensureCSS();
    if (!tipEl) {
        tipEl = document.createElement("div");
        tipEl.className = "luna-tip";
        document.body.appendChild(tipEl);
    }
    tipEl.textContent = text;
    tipEl.style.display = "block";
    // Flip to the left of the cursor when the tooltip would leave the window.
    const w = tipEl.offsetWidth || 280;
    tipEl.style.left = (x + 16 + w > window.innerWidth ? x - w - 12 : x + 16) + "px";
    tipEl.style.top = Math.min(y + 14, window.innerHeight - (tipEl.offsetHeight || 40) - 8) + "px";
}

function hideTip() {
    if (tipEl) tipEl.style.display = "none";
}

/** Which output slot is the cursor over? pos is node-relative graph coords. */
function outputAt(node, pos) {
    if (!node.outputs?.length) return -1;
    for (let i = 0; i < node.outputs.length; i++) {
        const p = node.getConnectionPos?.(false, i);
        if (!p) continue;
        // getConnectionPos is absolute; pos is node-relative.
        const dx = pos[0] + node.pos[0] - p[0];
        const dy = pos[1] + node.pos[1] - p[1];
        if (Math.abs(dx) < 22 && Math.abs(dy) < 9) return i;
    }
    return -1;
}

/* -------------------------------------------------------------- the ⓘ icon */

function iconCentre(node) {
    const LG = globalThis.LiteGraph ?? globalThis.comfyAPI?.litegraph?.LiteGraph;
    const h = LG?.NODE_TITLE_HEIGHT ?? 30;
    return [node.size[0] - ICON_INSET, -h / 2];
}

/**
 * Add the help icon to every node whose class name passes `match`.
 *   registerLunaHelp(app, (name) => name.startsWith("Luna"), "MyPack.Help")
 */
export function registerLunaHelp(app, match, extensionName) {
    const wanted = typeof match === "function" ? match : (n) => match.includes(n);
    app.registerExtension({
        name: extensionName,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (!wanted(nodeData?.name)) return;

            const onDraw = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDraw?.apply(this, arguments);
                if (this.flags?.collapsed) return r;
                const [cx, cy] = iconCentre(this);
                ctx.save();
                ctx.beginPath();
                ctx.arc(cx, cy, ICON_R, 0, Math.PI * 2);
                ctx.strokeStyle = this._lunaHelpHover ? LUNA.accent : LUNA.muted;
                ctx.lineWidth = 1.4;
                ctx.stroke();
                ctx.fillStyle = this._lunaHelpHover ? LUNA.accent : LUNA.muted;
                ctx.font = "bold 10px 'Segoe UI', system-ui, sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("i", cx, cy + 0.5);
                ctx.restore();
                return r;
            };

            // pos is node-relative; the title bar sits at negative y.
            const hit = (node, pos) => {
                const [cx, cy] = iconCentre(node);
                return Math.hypot(pos[0] - cx, pos[1] - cy) <= ICON_R + 3;
            };

            const onDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function (e, pos) {
                if (hit(this, pos)) { openHelp(this); return true; }   // true = consumed
                return onDown?.apply(this, arguments);
            };

            const onMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function (e, pos) {
                const over = hit(this, pos);
                if (over !== this._lunaHelpHover) {
                    this._lunaHelpHover = over;
                    this.setDirtyCanvas(true, false);
                }
                // Output-socket tooltip, since ComfyUI does not draw one.
                const tips = this.constructor?.nodeData?.output_tooltips;
                if (tips?.length && !this.flags?.collapsed) {
                    const slot = outputAt(this, pos);
                    if (slot >= 0 && tips[slot]) showTip(tips[slot], e.clientX, e.clientY);
                    else hideTip();
                }
                return onMove?.apply(this, arguments);
            };

            // Leaving the node entirely must clear it — onMouseMove stops firing.
            const onLeave = nodeType.prototype.onMouseLeave;
            nodeType.prototype.onMouseLeave = function () {
                hideTip();
                this._lunaHelpHover = false;
                return onLeave?.apply(this, arguments);
            };
        },
    });
}
