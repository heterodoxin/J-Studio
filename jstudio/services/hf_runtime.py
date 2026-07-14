"""Real local Hugging Face decoder services for J Studio.

This is the backend boundary: PyTorch, Transformers, and jlens never cross into
the Qt package. A session without a fitted Jacobian checkpoint reports its real
residual readout as vanilla and keeps model interventions disabled.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from jstudio.domain import (
    DENSE_LENS_ESTIMATORS,
    ActivationSource,
    BackendKind,
    ConceptActivation,
    InterventionDraft,
    InterventionOperation,
    JLensFrame,
    LensFitState,
    LensFitStatus,
    ModelSessionSummary,
    RuleRecord,
    RuleTrigger,
    RunRecord,
    RunState,
    SessionCapabilities,
    SessionState,
)
from jstudio.rules.protocol import RuleEvaluationRequest, SandboxLimits
from jstudio.rules.sandbox import QuickJSSandbox
from jstudio.services.lens_fitting import (
    GPUCoordinator,
    ProgressiveLensController,
    RuntimeProgressiveFitter,
)
from jstudio.services.protocols import (
    GenerationRequest,
    JStudioServices,
    SliceRequest,
)
from jstudio.services.slice_runtime import SliceRendererService

DEFAULT_MODEL_ID = "heterodoxin/qwen3-8b-apostate"
LOGGER = logging.getLogger(__name__)


def _readout_values(z_score: float) -> tuple[float, float]:
    import math

    return z_score, 1.0 / (1.0 + math.exp(-z_score))


def _suffix_prefix_length(text: str, token: str) -> int:
    maximum = min(len(text), len(token) - 1)
    for size in range(maximum, 0, -1):
        if token.startswith(text[-size:]):
            return size
    return 0


def _sequence_count(tokens: tuple[int, ...], variants: tuple[tuple[int, ...], ...]) -> int:
    count = 0
    for variant in variants:
        if not variant:
            continue
        width = len(variant)
        count += sum(
            tokens[index : index + width] == variant
            for index in range(len(tokens) - width + 1)
        )
    return count


def _causal_token_effect(
    operation: str,
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
    source_variants: tuple[tuple[int, ...], ...],
    target_variants: tuple[tuple[int, ...], ...],
) -> tuple[bool, float]:
    """Judge a short generated-token probe without optimizing against logits."""
    width = max(len(baseline), len(candidate), 1)
    divergence = (
        sum(
            index >= len(baseline)
            or index >= len(candidate)
            or baseline[index] != candidate[index]
            for index in range(width)
        )
        / width
    )
    source_before = _sequence_count(baseline, source_variants)
    source_after = _sequence_count(candidate, source_variants)
    target_before = _sequence_count(baseline, target_variants)
    target_after = _sequence_count(candidate, target_variants)
    target_gain = target_after - target_before
    source_reduction = source_before - source_after
    score = float(divergence + max(target_gain, 0) + max(source_reduction, 0))
    if operation == "inject":
        target_token_ids = {
            token_id for variant in target_variants for token_id in variant
        }
        contextual_tokens = sum(
            token_id not in target_token_ids for token_id in candidate
        )
        return target_gain == 1 and contextual_tokens >= 2, score
    if operation == "replace":
        source_changed = source_reduction > 0 if source_before else divergence > 0
        return target_gain > 0 and source_changed, score
    if operation == "suppress":
        return (
            source_reduction > 0 if source_before else divergence > 0,
            score,
        )
    raise ValueError(f"unsupported phrase operation {operation!r}")


class ThinkingFilter:
    """Streaming filter that strips visible ``<think>...</think>`` blocks."""

    _open = "<think>"
    _close = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._hidden = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        output = []
        while self._buffer:
            if self._hidden:
                close = self._buffer.find(self._close)
                if close < 0:
                    keep = _suffix_prefix_length(self._buffer, self._close)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    return "".join(output)
                self._buffer = self._buffer[close + len(self._close) :]
                self._hidden = False
                continue
            open_at = self._buffer.find(self._open)
            if open_at < 0:
                keep = _suffix_prefix_length(self._buffer, self._open)
                emit = self._buffer[:-keep] if keep else self._buffer
                self._buffer = self._buffer[-keep:] if keep else ""
                output.append(emit)
                return "".join(output)
            output.append(self._buffer[:open_at])
            self._buffer = self._buffer[open_at + len(self._open) :]
            self._hidden = True
        return "".join(output)

    def flush(self) -> str:
        if self._hidden:
            self._buffer = ""
            return ""
        remaining = self._buffer
        self._buffer = ""
        return remaining


def default_lens_path(model_id: str) -> Path:
    override = os.environ.get("JSTUDIO_LENS")
    if override:
        return Path(override).expanduser()
    safe_name = model_id.replace("/", "--")
    workspace = Path(__file__).parents[3] / "lenses" / safe_name / "lens.pt"
    if workspace.exists():
        return workspace
    return Path.home() / ".cache" / "jstudio" / "lenses" / safe_name / "lens.pt"


class ModelRuntime(Protocol):
    model_id: str
    revision: str
    layer_count: int
    device: str
    precision: str
    lens_id: str | None
    calibrated: bool

    def read_activations(
        self,
        prompt: str,
        *,
        token_index: int,
        layers: tuple[int, ...] = (),
        max_concepts: int = 200,
    ) -> tuple[ConceptActivation, ...]: ...

    def stream(self, prompt: str, *, max_new_tokens: int, history: tuple = ()): ...

    def close(self) -> None: ...


class HFModelRuntime:
    """One BF16 causal decoder loaded on the primary CUDA or ROCm device."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        local_files_only: bool = True,
        max_new_tokens: int = 2048,
        lens_path: str | Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA or ROCm PyTorch device is required for the local model"
            )
        self._torch = torch
        self.coordinator = GPUCoordinator()
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda:0")
        self.model.eval()
        import inspect

        self._supports_logits_to_keep = (
            "logits_to_keep" in inspect.signature(self.model.forward).parameters
        )

        import jlens

        self.lens_model = jlens.from_hf(self.model, self.tokenizer)
        self.revision = getattr(self.model.config, "_commit_hash", None) or "local-cache"
        selected_lens = Path(lens_path) if lens_path else default_lens_path(model_id)
        self.stable_lens_path = selected_lens
        self.lens_path = selected_lens
        self.lens = None
        for candidate_path in self._candidate_lens_paths(selected_lens):
            if not candidate_path.is_file():
                continue
            try:
                candidate = jlens.JacobianLens.load(str(candidate_path))
                self._validate_lens(candidate)
            except ValueError as exc:
                if lens_path is not None and candidate_path == selected_lens:
                    raise
                del exc
            else:
                self.lens = candidate
                self.lens_path = candidate_path
                break
        self.layer_count = self.lens_model.n_layers
        backend = "ROCm" if torch.version.hip else "CUDA"
        self.device = f"{backend} 0 · {torch.cuda.get_device_name(0)}"
        self.precision = "BF16"
        self.lens_id = None
        self.calibrated = False
        if self.lens is not None:
            self._update_lens_identity()

    @staticmethod
    def _candidate_lens_paths(selected: Path) -> tuple[Path, ...]:
        root = selected.parent
        candidates = (
            selected,
            root / "stable.lens.pt",
            root / ".fit" / "stable.lens.pt",
        )
        return tuple(dict.fromkeys(candidates))

    def _validate_lens(self, lens, *, require_stable: bool = True) -> None:
        if lens.d_model != self.lens_model.d_model:
            raise ValueError("model and fitted Jacobian lens disagree on d_model")
        fitted_model = lens.metadata.get("model")
        if fitted_model and fitted_model != self.model_id:
            raise ValueError(
                f"lens was fitted for {fitted_model}, not requested model {self.model_id}"
            )
        estimator = lens.metadata.get("estimator")
        dense_family = estimator in DENSE_LENS_ESTIMATORS
        accepted = {"prompt-averaged-orthogonal-sketch-v2", *DENSE_LENS_ESTIMATORS}
        if estimator not in accepted:
            raise ValueError("lens uses an incompatible estimator version")
        expected_target = (
            self.lens_model.n_layers - 1
            if dense_family
            else self.lens_model.n_layers - 2
        )
        if lens.metadata.get("target_layer") != str(expected_target):
            raise ValueError("lens uses an incompatible Jacobian target layer")
        fitted_revision = lens.metadata.get("revision")
        if fitted_revision and fitted_revision != self.revision:
            raise ValueError("lens was fitted for a different model revision")
        quality_stage = lens.metadata.get("quality_stage")
        if require_stable and quality_stage != "Stable":
            raise ValueError("only a Stable fitted Jacobian lens can be used for J-space")
        if quality_stage not in {"Preview", "Stable"}:
            raise ValueError("lens is missing a Preview or Stable quality stage")
        if require_stable and dense_family:
            gate = lens.metadata.get("quality_gate_version")
            try:
                passed = int(lens.metadata.get("viewing_passed", "0"))
                total = int(lens.metadata.get("viewing_total", "0"))
            except ValueError:
                passed = total = 0
            if gate != "jspace-viewing-v2" or total <= 0 or passed != total:
                raise ValueError(
                    "Stable dense lens is missing a complete reference viewing gate"
                )
        elif require_stable:
            gate = lens.metadata.get("quality_gate_version")
            try:
                pass_at_10 = float(lens.metadata.get("fit_quality_pass_at_10", "nan"))
            except ValueError:
                pass_at_10 = float("nan")
            if gate != "jspace-v1" or pass_at_10 < 0.3:
                raise ValueError(
                    "Stable lens is missing the current J-space quality gate"
                )
        uncalibrated = [
            layer
            for layer in lens.source_layers
            if not lens.metric(layer).calibrated
        ]
        if require_stable and uncalibrated:
            raise ValueError(
                "Stable Jacobian lens is not fully calibrated; "
                f"uncalibrated layers: {uncalibrated[:8]}"
            )

    def _update_lens_identity(self) -> None:
        assert self.lens is not None
        if self.lens.metadata.get("estimator") in DENSE_LENS_ESTIMATORS:
            self.lens_id = f"dense-jacobian-n{self.lens.n_prompts}"
            passed = self.lens.metadata.get("viewing_passed")
            total = self.lens.metadata.get("viewing_total")
            if passed and total:
                self.lens_id += f" · viewing {passed}/{total}"
            shrinkage = self.lens.metadata.get("transport_shrinkage")
            if shrinkage:
                self.lens_id += f" · J×{shrinkage}"
        else:
            rank = self.lens.metadata.get("effective_rank") or self.lens.metadata.get(
                "sketch_rank", "unknown"
            )
            self.lens_id = f"sketched-jacobian-r{rank}"
        try:
            pass_at_10 = float(self.lens.metadata["fit_quality_pass_at_10"])
        except (KeyError, ValueError):
            pass
        else:
            if self.lens.metadata.get("estimator") not in DENSE_LENS_ESTIMATORS:
                self.lens_id += f" · pass@10 {pass_at_10:.2f}"
        self.calibrated = all(
            self.lens.metric(layer).calibrated for layer in self.lens.source_layers
        )

    def activate_lens(self, lens, path: str | Path, quality: str) -> None:
        lens.metadata.update(
            {
                "model": self.model_id,
                "revision": self.revision,
                "quality_stage": quality,
            }
        )
        self._validate_lens(lens, require_stable=quality == "Stable")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        lens.save(str(temporary))
        os.replace(temporary, path)
        with self.coordinator.exclusive("activate-lens"):
            self.lens = lens
            self.lens_path = path
            self._update_lens_identity()

    def _formatted_prompt(self, prompt: str, history: tuple = ()) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer directly. "
                    "Do not emit <think> blocks or hidden reasoning."
                ),
            },
        ]
        messages.extend({"role": role, "content": content} for role, content in history)
        messages.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, ValueError, TypeError):
            return prompt

    def inspection_context(
        self,
        prompt: str,
        output: str,
        *,
        history: tuple = (),
    ) -> str:
        """Build the exact causal transcript inspected by J-Lens."""
        formatted = self._formatted_prompt(prompt, history)
        if not output:
            return formatted
        separator = "\n\n" if formatted == prompt else ""
        return f"{formatted}{separator}{output}"

    @staticmethod
    def _verbalizable(term: str) -> bool:
        term = term.strip()
        return bool(
            term and len(term) <= 48 and any(character.isalnum() for character in term)
        )

    def read_activations(
        self,
        prompt: str,
        *,
        token_index: int,
        layers: tuple[int, ...] = (),
        max_concepts: int = 200,
    ) -> tuple[ConceptActivation, ...]:
        if self.lens is None:
            raise RuntimeError("J-Lens Preview is still fitting")
        selected_layers = (
            [layer for layer in layers if layer in self.lens.source_layers]
            if layers
            else self.lens.source_layers
        )
        if not selected_layers:
            raise ValueError(
                f"none of the selected layers are fitted; available: "
                f"{self.lens.source_layers}"
            )
        candidates: dict[str, tuple[float, int, int]] = {}
        with self.coordinator.exclusive("lens-readout"):
            lens_logits, _, _ = self.lens.apply(
                self.lens_model,
                prompt,
                layers=selected_layers,
                positions=None,
                max_seq_len=2048,
                use_jacobian=True,
            )
        for layer, position_logits in lens_logits.items():
            for position, logits in enumerate(position_logits):
                mean = logits.mean()
                std = logits.std().clamp_min(1e-6)
                values, ids = logits.topk(64)
                for value, token_id in zip(values, ids, strict=True):
                    term = self.tokenizer.decode([int(token_id)]).strip()
                    if not self._verbalizable(term):
                        continue
                    z_score = float(((value - mean) / std).item())
                    score, confidence = _readout_values(z_score)
                    previous = candidates.get(term)
                    if previous is None or score > previous[0]:
                        candidates[term] = (score, layer, position)
        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[
            :max_concepts
        ]
        return tuple(
            ConceptActivation(
                term=term,
                score=score,
                previous_score=None,
                confidence=_readout_values(score)[1],
                layer=layer,
                token_index=position,
                rank=rank,
                source=ActivationSource.OBSERVED,
            )
            for rank, (term, (score, layer, position)) in enumerate(ranked)
        )

    def stream(
        self, prompt: str, *, max_new_tokens: int | None = None, history: tuple = ()
    ):
        torch = self._torch
        text = self._formatted_prompt(prompt, history)
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.model.device)
        past_key_values = None
        current_ids = input_ids
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids or ())
        thinking_filter = ThinkingFilter()
        with torch.inference_mode():
            for _ in range(max_new_tokens or self.max_new_tokens):
                kwargs = {
                    "input_ids": current_ids,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                if self._supports_logits_to_keep:
                    kwargs["logits_to_keep"] = 1
                with self.coordinator.exclusive("generation"):
                    output = self.model(**kwargs)
                next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                past_key_values = output.past_key_values
                token_id = int(next_id.item())
                if token_id in eos_ids:
                    break
                visible = thinking_filter.feed(
                    self.tokenizer.decode([token_id], skip_special_tokens=True)
                )
                if visible:
                    yield visible
                current_ids = next_id
        tail = thinking_filter.flush()
        if tail:
            yield tail

    def _token_variants(self, text: str | None) -> tuple[tuple[int, ...], ...]:
        if not text:
            return ()
        variants = []
        for candidate in (text, f" {text}"):
            encoded = self.tokenizer(
                candidate, return_tensors="pt", add_special_tokens=False
            )
            token_ids = tuple(int(value) for value in encoded.input_ids[0].tolist())
            if token_ids and token_ids not in variants:
                variants.append(token_ids)
        return tuple(variants)

    def _causal_probe_ids(
        self, model_prompt: str, *, max_new_tokens: int = 12
    ) -> tuple[int, ...]:
        """Run a short deterministic completion used only as causal evidence."""
        torch = self._torch
        current_ids = self.tokenizer(model_prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        past_key_values = None
        generated = []
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or ())
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                kwargs = {
                    "input_ids": current_ids,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                if self._supports_logits_to_keep:
                    kwargs["logits_to_keep"] = 1
                output = self.model(**kwargs)
                next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                past_key_values = output.past_key_values
                token_id = int(next_id.item())
                if token_id in eos_ids:
                    break
                generated.append(token_id)
                current_ids = next_id
        return tuple(generated)

    def _make_phrase_effect_probe(self, model_prompt: str, draft):
        from jlens.hooks import (
            ResidualTransform,
            ResidualTransformEditor,
        )

        baseline = self._causal_probe_ids(model_prompt)
        source_variants = self._token_variants(draft.source_term)
        target_variants = self._token_variants(draft.target_term)
        target_steps = min((len(value) for value in target_variants), default=1)
        max_applications = (
            None
            if draft.duration == "generation"
            else max(draft.step_count or 1, target_steps)
            if draft.duration == "steps"
            else target_steps
        )

        def probe(operator_pairs, positions):
            editor = ResidualTransformEditor(
                self.lens_model.layers,
                [
                    ResidualTransform(
                        layer=layer,
                        positions=positions,
                        transform=operator.make_transform(),
                        max_applications=max_applications,
                    )
                    for layer, operator in operator_pairs
                ],
            )
            with editor:
                candidate = self._causal_probe_ids(model_prompt)
            return _causal_token_effect(
                draft.operation.value,
                baseline,
                candidate,
                source_variants,
                target_variants,
            )

        return probe

    def prepare_interventions(self, prompt: str, drafts, *, history: tuple = ()):
        if self.lens is None or not self.calibrated:
            raise RuntimeError("a calibrated Stable lens is required for interventions")
        import jlens

        model_prompt = self._formatted_prompt(prompt, history)
        editors = []
        results = []
        # Cap the search below the near-unembed layers where edits destroy coherence.
        coherent_ceiling = int(0.8 * (self.layer_count - 1))
        with self.coordinator.exclusive("intervention-search"):
            engine = jlens.InterventionEngine(self.lens_model, self.lens)
            for draft in drafts:
                try:
                    layers = [
                        layer
                        for layer in self.lens.source_layers
                        if draft.layer_start
                        <= layer
                        <= min(draft.layer_end, coherent_ceiling)
                    ]
                    if not layers:
                        raise ValueError("intervention range contains no fitted layers")
                    options = {
                        "layers": layers,
                        "positions": (-1,),
                        "maximum_scale": draft.strength,
                        "effect_probe": self._make_phrase_effect_probe(
                            model_prompt, draft
                        ),
                    }
                    if draft.operation.value == "inject":
                        result = engine.phrase_inject(
                            model_prompt, draft.target_term, **options
                        )
                    elif draft.operation.value == "replace":
                        result = engine.phrase_replace(
                            model_prompt,
                            draft.source_term,
                            draft.target_term,
                            **options,
                        )
                    else:
                        result = engine.phrase_suppress(
                            model_prompt, draft.source_term, **options
                        )
                    if not result.success:
                        editor = nullcontext()
                    elif draft.duration == "generation":
                        editor = engine.apply(result, once=False)
                    elif draft.duration == "steps":
                        editor = engine.apply(
                            result,
                            max_applications=max(
                                draft.step_count or 1,
                                len(result.trace.target_ids) or 1,
                            ),
                        )
                    else:
                        editor = engine.apply(
                            result,
                            max_applications=len(result.trace.target_ids) or 1,
                        )
                except Exception as exc:
                    trace = jlens.InterventionTrace(
                        operation=draft.operation.value,
                        target_ids=(),
                        source_ids=(),
                        experimental=True,
                        selected_layer=None,
                        selected_positions=(-1,),
                        selected_scale=0.0,
                        normalized_cost=0.0,
                        baseline_scores={},
                        after_scores={},
                        baseline_top_ids=(),
                        after_top_ids=(),
                        search_points=(),
                        warnings=("rule-preparation-failed",),
                    )
                    result = jlens.InterventionResult(
                        False,
                        trace,
                        None,
                        f"{type(exc).__name__}: {exc}",
                    )
                    editor = nullcontext()
                results.append(result)
                editors.append(editor)
        return tuple(editors), tuple(results)

    def close(self) -> None:
        self.model.to("cpu")
        self._torch.cuda.empty_cache()


class HFSessionService:
    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    def _summary(self) -> ModelSessionSummary:
        runtime = self._runtime
        lens_present = getattr(runtime, "lens", object()) is not None
        return ModelSessionSummary(
            session_id=f"local:{runtime.model_id}",
            model_id=runtime.model_id,
            revision=runtime.revision,
            lens_id=getattr(runtime, "lens_id", "test-jacobian-lens"),
            layer_count=runtime.layer_count,
            backend_kind=BackendKind.LOCAL,
            state=SessionState.READY,
            capabilities=SessionCapabilities(
                inspect=lens_present,
                generate=True,
                intervene=bool(getattr(runtime, "calibrated", False)),
                rules=True,
                strength_min=0.0,
                strength_max=16.0,
            ),
            display_name=runtime.model_id.rsplit("/", 1)[-1],
            device=runtime.device,
            precision=runtime.precision,
        )

    def list_sessions(self) -> tuple[ModelSessionSummary, ...]:
        return (self._summary(),)

    def open_session(self, session_id: str) -> ModelSessionSummary:
        session = self._summary()
        if session_id != session.session_id:
            raise KeyError(session_id)
        return session

    def refresh(self) -> tuple[ModelSessionSummary, ...]:
        return self.list_sessions()


class HFLensService:
    def __init__(
        self,
        runtime: ModelRuntime,
        controller: ProgressiveLensController | None = None,
    ) -> None:
        self._frames: dict[str, list[JLensFrame]] = {}
        self._lock = threading.Lock()
        self._slices = SliceRendererService(runtime)
        self._runtime = runtime
        self._controller = controller

    def record(self, frame: JLensFrame) -> None:
        with self._lock:
            self._frames.setdefault(frame.run_id, []).append(frame)

    def current_activations(self, run_id: str) -> tuple[ConceptActivation, ...]:
        frames = self.frames(run_id)
        return frames[-1].activations if frames else ()

    def frames(self, run_id: str) -> tuple[JLensFrame, ...]:
        with self._lock:
            return tuple(self._frames.get(run_id, ()))

    def request_slice(self, request: SliceRequest):
        if getattr(self._runtime, "lens", object()) is None:
            future = Future()
            detail = self.fit_status().detail or "J-Lens is fitting"
            future.set_exception(RuntimeError(detail))
            return future
        return self._slices.request_slice(request)

    def fit_status(self) -> LensFitStatus:
        if self._controller is not None:
            status = self._controller.status()
            if (
                status.state is not LensFitState.MISSING
                or getattr(self._runtime, "lens", object()) is None
            ):
                return status
        if getattr(self._runtime, "lens", object()) is None:
            return LensFitStatus(
                LensFitState.MISSING,
                "",
                0,
                0,
                detail="No compatible lens is active",
            )
        quality_stage = getattr(self._runtime.lens, "metadata", {}).get(
            "quality_stage", "Stable"
        )
        state = (
            LensFitState.PREVIEW
            if quality_stage == "Preview"
            else LensFitState.STABLE
        )
        return LensFitStatus(state, quality_stage, 1, 1, quality="active")

    def start_fit(self) -> None:
        if self._controller is not None:
            quality_stage = getattr(self._runtime.lens, "metadata", {}).get(
                "quality_stage"
            ) if getattr(self._runtime, "lens", None) is not None else None
            self._controller.start(force=quality_stage == "Stable")

    def cancel_fit(self) -> None:
        if self._controller is not None:
            self._controller.cancel()

    def subscribe_fit(self, callback):
        if self._controller is not None:
            return self._controller.subscribe(callback)
        return lambda: None

    def close(self) -> None:
        if self._controller is not None:
            self._controller.cancel()
            self._controller.join(timeout=5)
        self._slices.close()


@dataclass
class _RunControl:
    condition: threading.Condition
    paused: bool = False
    stopped: bool = False
    step_budget: int = 0


class HFGenerationService:
    def __init__(self, runtime: ModelRuntime, lens: HFLensService, rules=None) -> None:
        self._runtime = runtime
        self._lens = lens
        self._rules = rules or QuickJSSandbox()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jstudio-hf")
        self._controls: dict[str, _RunControl] = {}
        self._lock = threading.Lock()

    def start(self, request: GenerationRequest, sink) -> str:
        run = replace(
            RunRecord.create(prompt=request.prompt, mode=request.mode),
            intervention_ids=request.intervention_ids,
            rule_ids=request.rule_ids,
            state=RunState.RUNNING,
        )
        control = _RunControl(threading.Condition())
        with self._lock:
            self._controls[run.run_id] = control
        self._executor.submit(self._run, run, request, sink, control)
        return run.run_id

    def _rule_context(
        self,
        rule: RuleRecord,
        prompt: str,
        *,
        output_text: str = "",
        activations: tuple[ConceptActivation, ...] = (),
    ) -> dict:
        return {
            "event": {"type": rule.trigger.value, "sequence": 0},
            "model": {
                "id": self._runtime.model_id,
                "revision": self._runtime.revision,
                "layerCount": self._runtime.layer_count,
            },
            "lens": {"id": getattr(self._runtime, "lens_id", None)},
            "layer": {"index": max(0, self._runtime.layer_count - 2)},
            "token": {"index": 0, "text": ""},
            "generation": {"step": 0, "prompt": prompt, "outputText": output_text},
            "jspace": {
                "activations": [
                    {
                        "term": activation.term,
                        "score": activation.score,
                        "layer": activation.layer,
                        "position": activation.token_index,
                    }
                    for activation in activations
                ]
            },
            "stack": {"entries": []},
            "tags": {},
            "config": dict(rule.config),
        }

    def _layers_from_rule(self, value) -> tuple[int, int]:
        if value == "current":
            layer = max(0, self._runtime.layer_count - 2)
            return layer, layer
        if value == "all":
            return 0, max(0, self._runtime.layer_count - 1)
        return int(value["from"]), int(value["to"])

    def _duration_from_rule(self, value) -> tuple[str, int | None]:
        if isinstance(value, dict):
            return "steps", int(value["steps"])
        return str(value), None

    def _draft_from_rule_action(self, rule: RuleRecord, action) -> InterventionDraft | None:
        payload = action.payload
        if action.kind == "inject":
            layer_start, layer_end = self._layers_from_rule(payload["layers"])
            duration, step_count = self._duration_from_rule(payload["duration"])
            return InterventionDraft(
                InterventionOperation.INJECT,
                None,
                payload["term"],
                float(payload["strength"]),
                layer_start,
                layer_end,
                duration=duration,
                step_count=step_count,
                trigger=f"rule:{rule.rule_id}",
            )
        if action.kind == "replace":
            layer_start, layer_end = self._layers_from_rule(payload["layers"])
            duration, step_count = self._duration_from_rule(payload["duration"])
            return InterventionDraft(
                InterventionOperation.REPLACE,
                payload["matchTerm"],
                payload["replacementTerm"],
                float(payload["strength"]),
                layer_start,
                layer_end,
                duration=duration,
                step_count=step_count,
                match_mode=payload["matchMode"],
                trigger=f"rule:{rule.rule_id}",
            )
        if action.kind == "suppress":
            layer_start, layer_end = self._layers_from_rule(payload["layers"])
            duration, step_count = self._duration_from_rule(payload["duration"])
            return InterventionDraft(
                InterventionOperation.SUPPRESS,
                payload["term"],
                None,
                float(payload["strength"]),
                layer_start,
                layer_end,
                duration=duration,
                step_count=step_count,
                trigger=f"rule:{rule.rule_id}",
            )
        return None

    def _evaluate_rule_stack(self, request: GenerationRequest, sink):
        if not request.rule_records:
            return (), ()
        activations: tuple[ConceptActivation, ...] = ()
        if any(rule.trigger is RuleTrigger.JSPACE_FRAME for rule in request.rule_records):
            if getattr(self._runtime, "lens", object()) is not None:
                try:
                    activations = self._runtime.read_activations(
                        request.prompt,
                        token_index=0,
                        layers=request.read.layers,
                        max_concepts=request.read.max_concepts,
                    )
                except Exception:
                    activations = ()
        generated_ids = []
        generated_drafts = []
        for rule in sorted(request.rule_records, key=lambda item: item.priority):
            if rule.trigger not in {RuleTrigger.BEFORE_TOKEN, RuleTrigger.JSPACE_FRAME}:
                continue
            result = self._rules.evaluate(
                RuleEvaluationRequest(
                    source=rule.source,
                    trigger=rule.trigger.value,
                    context=self._rule_context(
                        rule,
                        request.prompt,
                        activations=activations,
                    ),
                    layer_count=self._runtime.layer_count,
                    limits=SandboxLimits(),
                )
            )
            if not result.success:
                sink.on_intervention(rule.rule_id, "failed", result.error)
                continue
            for index, action in enumerate(result.actions):
                draft = self._draft_from_rule_action(rule, action)
                if draft is None:
                    if action.kind == "log":
                        sink.on_intervention(
                            rule.rule_id,
                            "applied",
                            str(action.payload.get("message", "rule log")),
                        )
                    elif action.kind == "stop":
                        raise RuntimeError(action.payload["reason"])
                    continue
                generated_ids.append(f"{rule.rule_id}:{action.kind}:{index}")
                generated_drafts.append(draft)
        return tuple(generated_ids), tuple(generated_drafts)

    def _run(
        self,
        run: RunRecord,
        request: GenerationRequest,
        sink,
        control: _RunControl,
    ) -> None:
        try:
            request_started = time.perf_counter()
            sink.on_started(run)
            chunks: list[str] = []
            token_times: list[float] = []
            editors = ()
            rule_intervention_ids, rule_intervention_drafts = self._evaluate_rule_stack(
                request, sink
            )
            intervention_ids = (*request.intervention_ids, *rule_intervention_ids)
            intervention_drafts = (
                *request.intervention_drafts,
                *rule_intervention_drafts,
            )
            generation_context = (
                self._runtime.coordinator.generation()
                if hasattr(self._runtime, "coordinator")
                and hasattr(self._runtime.coordinator, "generation")
                else nullcontext()
            )
            with generation_context:
                if intervention_drafts:
                    if not hasattr(self._runtime, "prepare_interventions"):
                        raise RuntimeError("runtime does not support interventions")
                    editors, results = self._runtime.prepare_interventions(
                        run.prompt,
                        intervention_drafts,
                        history=request.history,
                    )
                    for intervention_id, result in zip(
                        intervention_ids, results, strict=True
                    ):
                        sink.on_intervention(
                            intervention_id,
                            "applied" if result.success else "failed",
                            result.message,
                        )
                with ExitStack() as stack:
                    for editor in editors:
                        stack.enter_context(editor)
                    for chunk in self._runtime.stream(
                        run.prompt,
                        max_new_tokens=request.read.max_new_tokens,
                        history=request.history,
                    ):
                        with control.condition:
                            while (
                                control.paused
                                and not control.step_budget
                                and not control.stopped
                            ):
                                control.condition.wait(timeout=0.2)
                            if control.stopped:
                                break
                            if control.paused and control.step_budget:
                                control.step_budget -= 1
                        chunks.append(chunk)
                        token_times.append(time.perf_counter())
                        sink.on_token(run.run_id, chunk, "".join(chunks))
            state = RunState.CANCELLED if control.stopped else RunState.COMPLETE
            ttft = token_times[0] - request_started if token_times else 0.0
            if len(token_times) > 1:
                decode_rate = (len(token_times) - 1) / (
                    max(token_times[-1] - token_times[0], 1e-9)
                )
            elif token_times:
                decode_rate = 1.0 / max(token_times[0] - request_started, 1e-9)
            else:
                decode_rate = 0.0
            output_text = "".join(chunks)
            if hasattr(self._runtime, "inspection_context"):
                inspection_text = self._runtime.inspection_context(
                    run.prompt,
                    output_text,
                    history=request.history,
                )
            else:
                inspection_text = run.prompt + (
                    "\n\n" + output_text if output_text else ""
                )
            finished = replace(
                run.with_state(state, output_text=output_text),
                ttft_seconds=ttft,
                decode_tokens_per_second=decode_rate,
                inspection_text=inspection_text,
            )
            sink.on_finished(finished)
            if control.stopped or getattr(self._runtime, "lens", object()) is None:
                return
            try:
                activations = self._runtime.read_activations(
                    inspection_text,
                    token_index=0,
                    layers=request.read.layers,
                    max_concepts=request.read.max_concepts,
                )
            except Exception:
                return
            positions: dict[int, list[ConceptActivation]] = {}
            for activation in activations:
                positions.setdefault(activation.token_index, []).append(activation)
            for sequence, (position, values) in enumerate(sorted(positions.items())):
                frame = JLensFrame(
                    run_id=run.run_id,
                    sequence=sequence,
                    token_index=position,
                    token_text=f"prompt:{position}",
                    layer_count=self._runtime.layer_count,
                    activations=tuple(values),
                    timestamp=datetime.now(UTC).isoformat(),
                    interventions_active=intervention_ids,
                )
                self._lens.record(frame)
                sink.on_frame(frame)
        except Exception as exc:
            LOGGER.exception("Local Qwen generation failed for run %s", run.run_id)
            sink.on_error(run.run_id, "Local Qwen generation failed", repr(exc))
        finally:
            with self._lock:
                self._controls.pop(run.run_id, None)

    def _control(self, run_id: str) -> _RunControl:
        with self._lock:
            control = self._controls.get(run_id)
        if control is None:
            raise KeyError(run_id)
        return control

    def pause(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.paused = True

    def resume(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.paused = False
            control.condition.notify_all()

    def next_token(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.paused = True
            control.step_budget += 1
            control.condition.notify_all()

    def stop(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.stopped = True
            control.condition.notify_all()

    def close(self) -> None:
        with self._lock:
            controls = tuple(self._controls.values())
        for control in controls:
            with control.condition:
                control.stopped = True
                control.condition.notify_all()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._lens.close()
        self._runtime.close()


class RuntimeInterventionService:
    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    def preview(self, session_id, draft):
        runtime = self._runtime
        if not getattr(runtime, "calibrated", False):
            return False, (
                "A matching calibrated Jacobian lens is required before model "
                "interventions can be armed."
            )
        if draft.strength <= 0:
            return False, "Maximum strength must be greater than zero"
        lens = runtime.lens
        layers = [
            layer
            for layer in lens.source_layers
            if draft.layer_start <= layer <= draft.layer_end
        ]
        if not layers:
            return False, "Layer range contains no fitted J-Lens layers"
        import jlens

        resolver = jlens.ConceptResolver(runtime.tokenizer)
        try:
            if draft.target_term:
                target = resolver.resolve(draft.target_term)
                if draft.operation.value == "replace" and target.experimental:
                    return False, "Replacement requires a single-token target"
            if draft.source_term:
                source = resolver.resolve(draft.source_term)
                if draft.operation.value != "inject" and source.experimental:
                    return False, "Replacement/suppression requires a single token"
        except ValueError as exc:
            return False, str(exc)
        return True, (
            f"Will search the minimum effective scale up to {draft.strength:g} "
            f"across {len(layers)} fitted layers"
        )

    def bake(self, session_id, drafts, path):
        runtime = self._runtime
        if not getattr(runtime, "calibrated", False) or runtime.lens is None:
            raise ValueError("A calibrated J-space lens is required for baking")
        if not drafts:
            raise ValueError("Enable at least one intervention before baking")
        import jlens

        rules = []
        for draft in drafts:
            layers = tuple(
                layer
                for layer in runtime.lens.source_layers
                if draft.layer_start <= layer <= draft.layer_end
            )
            if not layers:
                raise ValueError(
                    "An enabled intervention has no fitted layers in its range"
                )
            rules.append(
                jlens.ProjectionBakeRule(
                    draft.operation.value,
                    source=draft.source_term,
                    target=draft.target_term,
                    strength=max(0.01, min(1.0, draft.strength / 16.0)),
                    layers=layers,
                )
            )
        baked = jlens.bake_projection(runtime.lens_model, runtime.lens, tuple(rules))
        return jlens.save_projection_bake(path, baked)


def services_for_runtime(runtime: ModelRuntime, *, rules=None) -> JStudioServices:
    controller = None
    if hasattr(runtime, "lens_model"):
        cache = runtime.lens_path.parent
        controller = ProgressiveLensController(
            runtime,
            RuntimeProgressiveFitter(runtime, cache / ".fit"),
            cache,
        )
    lens = HFLensService(runtime, controller)
    services = JStudioServices(
        sessions=HFSessionService(runtime),
        generation=HFGenerationService(runtime, lens, rules),
        lens=lens,
        interventions=RuntimeInterventionService(runtime),
        rules=rules or QuickJSSandbox(),
    )
    if controller is not None and getattr(runtime, "lens", object()) is None:
        controller.start()
    return services


def create_hf_services(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    local_files_only: bool = True,
    lens_path: str | Path | None = None,
) -> JStudioServices:
    rules = QuickJSSandbox()
    return services_for_runtime(
        HFModelRuntime(model_id, local_files_only=local_files_only, lens_path=lens_path),
        rules=rules,
    )
