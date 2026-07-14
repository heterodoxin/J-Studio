# jlens — Jacobian lens

> **Reference implementation.** Not maintained and not accepting contributions.

Companion code for [**Verbalizable Representations Form a Global Workspace in
Language Models**](https://transformer-circuits.pub/2026/workspace/index.html).

The Jacobian lens reads out what an internal activation is disposed to make the
model say. It linearly transports a residual-stream vector at any layer and
position into the final-layer basis, then decodes it with the model's own
unembedding into a ranked list of vocabulary tokens.

The transport is the average input–output Jacobian over a text corpus:

```
lens_l(h) = unembed( J_l @ h ), J_l = E[∂h_final / ∂h_l]
```

The expectation is over prompts, source positions, and all current-and-future
target positions in a generic web-text corpus; the precise estimator
(cotangents summed over target positions, then averaged over source positions)
is documented in the [`jlens.fitting`](jlens/fitting.py) module docstring.

This repo fits the lens on open-weights decoder transformers, applies it, and
renders the interactive layer × position view shown below. Examples use Qwen;
other HuggingFace decoders adapt cleanly.

![Slice visualisation: ASCII-face example](assets/slice_vis.png)

*The ASCII-face example: selecting the `^` (nose) position shows the lens
reading out "nose" at mid layers, although the word never appears in the
prompt.*

## Install

```bash
pip install -e .
```

## Usage

### Apply

To apply a pre-fitted lens:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (§9.3); ~100 prompts is usable. This is a
reference implementation and is not optimized; fitting time is dominated by
the model's own backward pass. Parallelize by running `fit()` on disjoint
slices and combining with `JacobianLens.merge()`.

### Inject, suppress, and replace concepts

The intervention API edits decoder residuals in J-lens coordinates. It solves
for a minimum-cost perturbation under the fitted residual covariance, verifies
the result with real model forward passes, and uses bounded bracketing plus
bisection to return the weakest passing strength.

```python
import jlens

# Calibration is inference-only and reusable. Existing lens files also work,
# but fall back to explicitly reported Euclidean/identity geometry.
lens = jlens.calibrate_geometry(
    model,
    lens,
    prompts=my_calibration_prompts,
    rank=16,
    shrinkage=0.05,
)

engine = jlens.InterventionEngine(model, lens)

injected = engine.inject(
    "Fact: The currency used in the country shaped like a boot is",
    "lira",
    top_k=5,
)
print(injected.success, injected.trace.selected_scale)

# Apply the calibrated edit to the next prefill pass. The one-shot default is
# safe for cached autoregressive generation.
with engine.apply(injected):
    output_ids = hf.generate(**tok("Continue:", return_tensors="pt").to(hf.device))

replaced = engine.replace(
    "Think of a field sport, then name it:",
    source="soccer",
    target="rugby",
)
print(replaced.trace.normalized_cost, replaced.trace.after_scores)

suppressed = engine.suppress(
    "Think of a field sport, then name it:", "soccer", top_k=10
)
```

Words that map to one tokenizer token are the reliable path. Arbitrary text is
accepted for injection as an experimental joint-token intervention; the trace
records the tokenizer resolution and sequence log-probability check.
Replacement and suppression currently require one token per concept.

Search is deliberately bounded. If no candidate layer and position meets the
criterion below `maximum_scale`, the result has `success=False` and contains
the best observed trace instead of silently applying a stronger edit.

The first intervention release targets Hugging Face decoder-only causal
language models through `jlens.from_hf`. Ollama and other inference servers do
not expose the residual activations required for this operation.

## Walkthrough

[`walkthrough.ipynb`](walkthrough.ipynb) is the end-to-end notebook: load a
model, load (or fit) a lens, apply it at a few layers, and render a slice page
like the one above.

Reading a slice page:

- Each cell shows the lens top-1 word at that (position, layer); the
  superscript is its rank over the full vocabulary.
- Click a cell to select a (position, layer) and pin its top-1 token; pinned
  tokens get rank-tracking charts and a rank heatmap.
- The bottom row (`L = n_layers − 1`) is the model's actual output.

Pass an `InterventionTrace` to `build_page(..., intervention_trace=trace)` to
show the selected operation, layer, position, scale, normalized cost, and
warnings above the existing slice view. Fetch-mode pages also write an atomic
`intervention.json` sidecar for later UIs.

## License and data

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

The replication and lens-eval prompt sets in [`data/`](data/) are synthetic,
authored by Anthropic, and released under the same Apache License 2.0 as the
code. See the READMEs in [`data/experiments/`](data/experiments/) and
[`data/evaluations/`](data/evaluations/) for what each set contains.

The slice-vis pages use [d3](https://github.com/d3/d3) (ISC license), loaded
from the jsDelivr CDN with subresource integrity or inlined into
self-contained pages.

No model weights or text corpora are bundled; models and datasets downloaded
at run time are subject to their own licenses.
