# Intervention Context Actions and Coherent Injection Design

## Goal

Make intervention rows directly manageable by right-click and prevent a semantic
injection from collapsing a completion into only the injected phrase.

## Interaction design

Right-clicking a row selects it and opens a compact menu with Enable/Disable, Edit,
Duplicate, Preview, Move Up, Move Down, and Remove. Multi-row selection supports
enable/disable, duplicate, and remove; actions that require one row are disabled for
multi-selection. Every mutation updates the project, table model, dirty flag, and
stack arming state through the existing `MainReadWorkspace` boundary.

Edit reuses the existing intervention editor with every draft field pre-populated and
replaces the entry by stable intervention ID. Duplicate creates new disarmed entries
immediately after the selection. Reordering changes execution order because stack
order is causal order.

## Influence routing

The Found Concepts context menu emits an influence request carrying the clicked term.
The shell opens the existing Influence Trace window, places that term in Seed Term,
and refreshes the graph with the seed first. The action therefore has an immediate,
visible result while retaining the window's existing observational-versus-causal
disclaimer.

## Injection acceptance

Minimum-effective-strength search remains J-space-only and evaluates generated token
sequences, never logits. An inject probe passes only when the target phrase gains at
least one occurrence and the candidate contains at least two tokens outside the
target token set. This rejects degenerate outputs such as `banana` or repeated target
tokens while allowing short natural continuations such as `Bananas are fruit`.
Failure to find a qualifying dose remains explicit; there is no fallback steering
path.

## Validation

Qt tests cover menu contents, selection-sensitive state, project mutations, editor
prefill, and influence-window routing. Runtime tests prove a target-only completion
fails and a mixed target-plus-context completion passes. The full app and lens test
suites, lint, offscreen smoke launch, and a real Qwen injection probe are run before
release.
