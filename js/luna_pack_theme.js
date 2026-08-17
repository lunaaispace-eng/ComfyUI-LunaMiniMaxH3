// Applies the Luna house theme to this pack's nodes.
//
// Roles come from the section tints in luna_theme.mjs. One line per node — this
// file is the per-pack map; the three modules it imports are the shared kit,
// copied verbatim from ComfyUI-SaveSimple (same author, same Apache-2.0 licence).
// Fix a bug in one of them and copy it across; there is no cross-pack import in
// ComfyUI's frontend.

import { app } from "../../scripts/app.js";
import { registerLunaTheme } from "./luna_theme.mjs";
import { registerLunaHelp } from "./luna_help.mjs";
import { registerLunaCollapse } from "./luna_collapse.mjs";

// Title-bar tinting stays off here for the same reason as in SaveSimple — Peti
// prefers ComfyUI's default red header with the widget's own palette inside.
// Uncommenting one call turns it back on.
//
// registerLunaTheme(app, {
//     LunaMiniMaxH3Canvas: "input",
// }, "LunaMiniMaxH3.Theme");

// The ⓘ on the title bar, built from each node's Python DESCRIPTION and its input
// tooltips, so there is no separate help file to keep up to date.
registerLunaHelp(app, ["LunaMiniMaxH3Canvas", "LunaH3ConcatAVLatent"], "LunaMiniMaxH3.Help");

// The chevron that folds settings away. Nothing uses it yet: the Canvas is all
// dials with a live readout, and the concat node has two sockets and a toggle.
// It is registered with an empty list so adding a node here is one word.
registerLunaCollapse(app, [], "LunaMiniMaxH3.Collapse");
