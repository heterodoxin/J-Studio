# J Studio

## What is this project

J Studio is a native desktop workbench for inspecting the internal J-space of decoder-only language models and applying calibrated word-level interventions on a local Hugging Face model running on a CUDA or ROCm GPU. It reads a fitted Jacobian lens across layers and token positions, renders the readout as an interactive layer-by-position slice, and arms inject, replace, and suppress operations that are verified against the model's own generation before they are applied. PyTorch, Transformers, and the lens backend sit behind a backend-neutral service boundary, and a deterministic demo mode and fail-closed QuickJS rules sandbox are included.

## Install

```bash
git clone https://github.com/heterodoxin/J-Studio.git
cd J-Studio
pip install -e .
```

The command above installs the desktop application and its Qt and rules-sandbox dependencies. The local model backend additionally requires a GPU build of PyTorch, Transformers, and the Jacobian lens library. Install the PyTorch build that matches your GPU.

On NVIDIA (CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

On AMD (ROCm):

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

Then install the remaining backend dependencies:

```bash
pip install transformers
pip install git+https://github.com/anthropics/jacobian-lens.git
```

Launch the workbench:

```bash
j-studio
```

Run without a model on deterministic demo data:

```bash
j-studio --demo
```

## Why it's better

The lens readout uses a dense-family Jacobian transport rather than a low-rank random sketch, so per-position token ranks reflect the fitted average Jacobian instead of amplified projection noise. Interventions are confirmed on the real generation path: an inject search returns the smallest steering strength whose short greedy probe contains the target concept while output stays coherent, and an edit that does not change generation is reported as not applied rather than silently dropped. The GUI package holds no PyTorch or Transformers imports; all model execution crosses a single service boundary, and a shared-GPU coordinator serializes fitting, readout, and generation against one loaded model.

## Why should I care

You can see the token a model is disposed to emit at any layer and position, pin those readouts, and watch their ranks across the network, all against a local model on your own hardware. Word-level interventions let you steer generation toward a chosen concept and observe the result token by token, with a provenance badge that states which lens is active and whether its readout is trustworthy. A background auto-fit produces a usable lens for a newly loaded model with a live progress bar and time estimate, so a model with no fitted lens becomes inspectable without leaving the application.

## Advanced

The auto-fit estimator projects each per-prompt input-output Jacobian onto the subspace spanned by the model's real final-layer residuals, accumulates the correction in two independent prompt halves, and keeps only singular directions that reproduce across the split via a cross-validated singular-value shrinkage. It fits the deep band of source layers, targets the final transport layer, calibrates residual-covariance geometry for intervention cost, and reports held-out pass@10 without gating on it. J-space injection steers generation by adding the lens readout covector for the target token, unit-normalized and scaled to a fraction of the residual norm, persistently at every generated position; the layer and strength are searched per model and per intervention, preferring the gentlest setting that produces a coherent steer. Streaming output strips `<think>` reasoning blocks incrementally, and the CUDA or ROCm backend loads one BF16 decoder whose residual blocks are located by trying known Hugging Face layouts in order.

## Disclaimer

Developed with assistance from Claude (Anthropic).
