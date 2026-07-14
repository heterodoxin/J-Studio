# Real Qwen ROCm Runtime Design

## Objective

J Studio launches against a real local decoder model by default. The deterministic
fake backend remains available only through `--demo`. The supported local reference
session is the already-cached `heterodoxin/qwen3-8b-apostate` checkpoint, loaded in
BF16 on ROCm through system Python 3.11.

## Runtime boundary

`jstudio.services.hf_runtime` is the only J Studio module allowed to import PyTorch,
Transformers, or `jlens`. It owns model loading, generation, residual capture, token
decoding, cancellation, and frame storage. Qt views continue to consume the existing
service protocols and never access model objects.

Model loading is explicit and fail-fast. The default launcher uses local Hugging Face
cache files and reports a precise startup error if the checkpoint, ROCm device, or
dependencies are unavailable. `--model` selects another compatible decoder checkpoint;
`--demo` is the only path to fixed fake data.

## Readout semantics

Until a matching fitted Jacobian lens is present, the session is labeled
`Live vanilla readout (uncalibrated)`. Found concepts come from real residual states:
selected layer residuals are passed through the model's final normalization and
unembedding, decoded, filtered, and ranked. They are never called calibrated J-space.
Live generation is real Qwen output. Intervention application remains disabled in an
uncalibrated session; drafts are still permitted.

When a compatible lens checkpoint is supplied later, the same runtime substitutes
Jacobian transport and enables the calibrated intervention engine without changing UI
interfaces.

## Resource behavior

The model is loaded once in BF16 on the primary ROCm device. Generation runs in the
existing non-Qt executor, supports pause, one-token stepping, resume, and stop, and
emits bounded frame updates. Default generation is deterministic greedy decoding with
128 new tokens. Closing J Studio stops active runs and releases executor resources.

## Verification

Unit tests use injected lightweight runtime doubles and prove that the launcher defaults
to real services, `--demo` is explicit, frames are derived from runtime readouts, and
uncalibrated sessions cannot arm interventions. A manual smoke test loads the cached
8B checkpoint, prompts `make me an ascii cat`, and verifies generated text and concepts
are not the deterministic prompt-injection fixture.
