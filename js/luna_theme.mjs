// Luna house theme — shared node colouring for the Luna ComfyUI packs.
//
// SOURCE OF TRUTH: ComfyUI-LunaRunner/web/style.css. This file is a copy so each
// pack stays self-contained (they are separate repos and a pack must work alone).
// If the palette changes there, change it here too.
//
//   --accent  #e0a458  warm amber      --text  #e7e5df    --muted  #8b8e99
//   --bg      #14151a  --panel #1c1e25 --border #2e313c   --radius 8px
//   section tints:  input #7bb47b · processing #7da4c8 · output #d98a6a
//
// The section tints are the useful part: node CATEGORY becomes readable at a
// glance across a graph. Title bars get a darkened tint so white text stays
// legible; bodies share one neutral panel colour so a graph does not turn into a
// patchwork.

export const LUNA = {
    accent: "#e0a458",
    accentSoft: "#e0a45822",
    bg: "#14151a",
    panel: "#1c1e25",
    border: "#2e313c",
    text: "#e7e5df",
    muted: "#8b8e99",
    danger: "#c0564a",
    radius: 8,
    input: "#7bb47b",
    processing: "#7da4c8",
    output: "#d98a6a",
};

// Title bars: the tint at roughly a third of its brightness. Picked so the
// default white title text keeps contrast rather than glowing. These are only
// the DEFAULTS — each is overridable from ComfyUI's Settings panel, so changing
// the scheme never means editing this file.
export const TITLE_DEFAULTS = {
    input: "#33513a",
    processing: "#33455a",
    output: "#5a3f33",
    agent: "#4a3a5a",   // browser / LLM nodes — outside the input/output triplet
};

const SETTING = {
    enabled: "Luna.Theme.Enabled",
    body: "Luna.Theme.BodyColor",
    input: "Luna.Theme.InputColor",
    processing: "Luna.Theme.ProcessingColor",
    output: "Luna.Theme.OutputColor",
    agent: "Luna.Theme.AgentColor",
};

function setting(app, id, fallback) {
    try {
        const v = app?.ui?.settings?.getSettingValue?.(id);
        return v === undefined || v === null || v === "" ? fallback : v;
    } catch {
        return fallback;
    }
}

const titleFor = (app, role) => setting(app, SETTING[role], TITLE_DEFAULTS[role] ?? TITLE_DEFAULTS.processing);

/** Repaint every themed node on screen — used when a colour setting changes.
 *
 * Deferred by a frame on purpose: onChange fires BEFORE the settings store has
 * committed the new value, so reading it immediately gives the previous one and
 * every change lands one step late. */
function recolourSoon(app) {
    requestAnimationFrame(() => recolourAll(app));
}

function recolourAll(app) {
    for (const n of app?.graph?._nodes ?? []) {
        if (!n._lunaRole) continue;
        n._lunaThemed = false;
        applyLunaTheme(n, n._lunaRole, app);
    }
    app?.graph?.setDirtyCanvas(true, true);
}

/** Register the colour settings once per session. Call from any pack; guarded. */
export function registerLunaSettings(app) {
    if (globalThis.__lunaThemeSettings) return;
    globalThis.__lunaThemeSettings = true;
    const colour = (id, name, fallback) => ({
        id, name, type: "text", defaultValue: fallback,
        category: ["Luna", "Node colours", name],
        tooltip: "Hex colour, e.g. #33513a. Blank restores the default.",
        onChange: () => recolourSoon(app),
    });
    app.registerExtension({
        name: "Luna.Theme.Settings",
        settings: [
            { id: SETTING.enabled, name: "Colour node title bars", type: "boolean",
              defaultValue: true, category: ["Luna", "Node colours", "Enabled"],
              onChange: () => recolourSoon(app) },
            colour(SETTING.input, "Input nodes", TITLE_DEFAULTS.input),
            colour(SETTING.processing, "Processing nodes", TITLE_DEFAULTS.processing),
            colour(SETTING.output, "Output nodes", TITLE_DEFAULTS.output),
            colour(SETTING.agent, "Agent nodes", TITLE_DEFAULTS.agent),
            colour(SETTING.body, "Node body", LUNA.panel),
        ],
    });
}

/**
 * Paint one node by role: "input" | "processing" | "output" | "agent".
 * Respects a colour the user set by hand — recolouring someone's deliberate
 * choice on every reload would be worse than having no theme at all.
 */
export function applyLunaTheme(node, role = "processing", app = globalThis.app) {
    if (node._lunaThemed) return;
    node._lunaRole = role;
    if (!setting(app, SETTING.enabled, true)) {
        // Theming turned off: hand the node back to ComfyUI's own defaults rather
        // than leaving it stuck on whatever Luna painted it last time.
        node.color = undefined;
        node.bgcolor = undefined;
        return;
    }
    node.color = titleFor(app, role);
    node.bgcolor = setting(app, SETTING.body, LUNA.panel);
    node._lunaThemed = true;
}

/**
 * Wire a whole pack in one call: map node class names to roles, then let this
 * colour them as they are created.
 *
 *   registerLunaTheme(app, { LunaAssetLoader: "input", SaveImageSimple: "output" });
 */
export function registerLunaTheme(app, roles, extensionName) {
    registerLunaSettings(app);
    app.registerExtension({
        name: extensionName,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            const role = roles[nodeData?.name];
            if (!role) return;
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onCreated?.apply(this, arguments);
                applyLunaTheme(this, role, app);
                return r;
            };
            // Nodes restored from a saved workflow do not run onNodeCreated.
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure?.apply(this, arguments);
                // A colour stored in the workflow is the user's; leave it alone.
                if (!info?.color) applyLunaTheme(this, role, app);
                return r;
            };
        },
    });
}
