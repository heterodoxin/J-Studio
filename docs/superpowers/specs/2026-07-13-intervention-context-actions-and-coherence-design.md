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
sequences, never logits. An inject probe passes only when the target phrase gains
exactly one occurrence and the candidate contains at least two tokens outside the
target token set. This rejects degenerate outputs such as `banana`, target repetition
after an otherwise normal prefix, or repeated target tokens while allowing short
natural continuations such as `Bananas are fruit`.
Failure to find a qualifying dose remains explicit; there is no fallback steering
path.

### Baseline trajectory preservation

Target presence is necessary but not sufficient. A successful intervention must
also preserve the unmodified completion's response trajectory. Preservation is
measured only from generated token IDs: remove the intended injected phrase from
the candidate, then measure ordered token overlap against the baseline completion.
The candidate must retain a majority of the baseline tokens in order. A candidate
that merely prefixes or suffixes the target to an otherwise identical baseline is
also rejected because that is concatenation, not a contextual change.

For injection, the target must occur exactly once and the preserved candidate must
retain the baseline trajectory. For replacement, comparison excludes the source
from the baseline and the target from the candidate. For suppression, comparison
excludes the suppressed source from the baseline. These checks judge generated
behavior without reading or optimizing logits.

The minimum-effective-strength search selects the first dose that passes both the
causal-direction check and the preservation check. If no dose passes, the
intervention fails closed and the baseline response is generated unchanged. The
runtime never substitutes a logit bias, prompt rewrite, or other fallback.

### Workspace-site application

The default Next Token injection is a context intervention, not an output-boundary
edit. J Studio locates the exact current user-turn token span inside the formatted
chat prompt and applies each layer-specific J-lens vector once across that span. It
never silently falls back to the final assistant-prefill token when span matching
fails.

Eligible layers are constrained to 38–75% of model depth, intersected with the
fitted lens layers and the user's explicit layer range. This targets the
workspace-like intermediate regime while avoiding late motor/output layers that
turn a concept vector into a forced first token. The relative band generalizes to
decoder depths without model-specific absolute layer numbers.

Multi-token injection targets use the normalized centroid of their token directions
at each user-turn position. They do not cycle token fragments over prompt positions.
Replace and Suppress remain localized coordinate edits at the response boundary;
their purpose is an explicit output-coordinate change rather than context priming.
Explicit Steps and Generation durations retain ordered per-generation scheduling;
their generated-sequence probe must still pass the same coherence checks.

## Validation

Qt tests cover menu contents, selection-sensitive state, project mutations, editor
prefill, and influence-window routing. Runtime tests prove target-only,
repeated-target, target-concatenation, and trajectory-derailing completions fail
while a contextual target insertion that retains the baseline trajectory passes.
The full app and lens test suites, lint, offscreen smoke launch, and a real Qwen
injection probe are run before release.
