# J Studio Modern Workbench UI Design

**Status:** Approved for implementation  
**Date:** 2026-07-13  
**Scope:** Visual hierarchy, progressive disclosure, shared theme, and J-Lens presentation

## Goal

Make J Studio clearer, calmer, and more modern without discarding the fast
Cheat Engine workflow that shaped the product. The redesign preserves the four
shared workspaces, model/session identity, scan-and-refine loop, found-concept
table, and intervention stack. It removes visual noise by making primary actions
obvious and moving infrequent controls behind purposeful disclosure.

The application remains an expert desktop research tool. “Modern” means strong
hierarchy, consistent spacing, readable states, and cohesive surfaces—not a
card-heavy web dashboard or a sparse consumer chat application.

## Approaches Considered

1. **Cosmetic refresh only.** Keep every control in place and add colors,
   rounded corners, and spacing. This is low risk but does not solve clutter.
2. **Full dashboard redesign.** Replace the scanner layout with navigation rails,
   cards, and inspector panels. This looks contemporary but loses the direct
   scan/results/stack workflow and consumes too much horizontal space.
3. **Modern workbench hybrid — selected.** Preserve the functional regions and
   shortcuts while rebuilding their hierarchy, disclosure, and visual language.
   This directly satisfies both requested goals.

## Visual Direction

The default presentation is a graphite dark theme with a restrained violet
accent derived from the J Studio logo. Surfaces use three elevations: window,
panel, and data/editor. Borders are low contrast; selection and focus are
high-contrast violet. Inject, replace, and suppress retain distinct semantic
colors, always paired with labels.

The UI uses the platform sans-serif font with a compact monospace face for
tokens, scores, layers, and code. Controls use 32–36 px heights, 6–10 px corner
radii, and an 8 px spacing rhythm. Headings are quiet but clearly separated from
metadata. Empty states explain the next useful action.

The theme is centralized in `jstudio/ui/theme.py`. Widgets receive semantic
object names or properties rather than one-off inline style sheets. Light and
system palettes remain supported, but the J Studio dark theme is the polished
default.

## Application Shell

The shell retains File/Edit/Model/Table/Tools/Help and Main/Chat/J-Lens/Rules.
The tab row becomes a modern workspace switcher with larger targets and a clear
active indicator. It does not become a navigation rail.

The session strip becomes a compact identity bar:

- logo/select action and model name are visually primary;
- revision, device, and precision become secondary metadata;
- backend, readiness, and lens quality become compact status pills;
- overflow retains infrequent session commands.

The status bar remains available for transient messages and errors but loses
heavy native borders.

## Main Workspace

The upper split remains Found Concepts on the left and prompt/read controls on
the right. The lower intervention stack remains shared across all tabs.

Changes:

- Each side is a named panel with consistent padding and a short supporting
  description or status.
- First Read is the primary violet action. Next and Undo are secondary actions.
- Prompt is visually dominant. Read Type and Concept Type form one compact row.
- `J-Space Scan Options` becomes a collapsed `Advanced scan` disclosure by
  default. Its existing fields and object identities remain intact for projects,
  tests, and keyboard workflows.
- The permanent bottom `Advanced Options` and `Rules` strip is removed. Rules
  remains a top workspace tab; advanced generation tools move to a small toolbar
  action near the scan controls.
- Found Concepts gets an instructive empty state and a compact footer for Add,
  Clear, J-Lens, and Model View.
- Intervention actions are grouped by intent: create operations on the left,
  validate/apply actions in the center, and destructive Clear on the right.
  Arm and Bake receive explicit state emphasis.

No controls are deleted from the behavior surface. Infrequent controls are
collapsed or relocated, and all existing signals remain connected.

## Chat and Rules

Chat receives a readable transcript canvas, clearly differentiated user and
assistant rows, a modern composer, and a compact shared-controls strip. Existing
streaming, regeneration, inspection, rule, and intervention behavior remains.

Rules keeps the list/editor/inspector split. The source editor is the dominant
surface. Rule settings become compact metadata above the API/Test inspector.
Problems, returned actions, and logs become a slimmer bottom drawer. Toolbar
actions use the same primary/secondary/danger hierarchy as Main.

## J-Lens Workspace

The J-Lens page is a first-class themed research surface, not a white notebook
embedded in a dark shell.

Native shell changes:

- a two-line header shows title/status on the left and lens/fit state plus
  Refresh/Export on the right;
- fit progress appears in a violet-accented inline panel;
- the web view has a subtle frame and fills the remaining workspace.

Rendered HTML changes in `jlens/data/slice_vis.html`:

- introduce CSS variables for graphite surfaces, text hierarchy, violet
  selection, semantic readout colors, grid lines, and plot backgrounds;
- preserve the original layer-by-position grid, prompt spatial layout, By Layer,
  By Position, heatmap, rank plots, hover, pinning, and intervention actions;
- restyle headers, controls, tooltips, selected rows/cells, scrollbars, and plots;
- use responsive grid proportions so the main readout stays legible at the
  enlarged J Studio launch size;
- clearly distinguish observational rank/readout marks from intervention state.

No J-Lens values, ranking math, event bridge, or serialization format changes.

## Accessibility and States

- Keyboard focus uses a visible violet outline.
- Disabled controls remain readable and visibly distinct.
- Color is never the only state signal.
- Minimum pointer targets remain 30 px.
- Text scaling tests continue to pass.
- Loading, empty, failed, preview, and stable-lens states have explicit text.
- Existing accessible names are preserved; new icon-only actions receive names
  and tooltips.

## Verification

Acceptance requires:

1. Existing backend-neutral UI behavior and all automated tests remain green.
2. New tests cover theme installation, progressive disclosure, semantic widget
   roles, and J-Lens theme tokens.
3. Offscreen screenshots cover Main, Chat, Rules, empty J-Lens, and populated
   J-Lens at 1101 × 888 or larger.
4. Screenshots are inspected for clipping, low contrast, unintended white
   surfaces, crowded toolbars, and inconsistent spacing.
5. `ruff`, full Pytest suites, package build, and editable install smoke pass.

## Release and Push Boundary

The current `origin` is `https://github.com/cheat-engine/cheat-engine.git`, which
is not an appropriate J Studio destination. The implementation will leave both
repositories cleanly committed and release-ready, but will not push J Studio to
that unrelated remote. A dedicated J Studio remote can be added and pushed as a
final non-destructive release step.
