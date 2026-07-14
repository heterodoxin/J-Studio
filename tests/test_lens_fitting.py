import threading
from types import SimpleNamespace

from jstudio.domain import LensFitState
from jstudio.services.lens_fitting import (
    GPUCoordinator,
    ProgressiveLensController,
    RuntimeProgressiveFitter,
)


def test_runtime_fitter_requires_all_reference_viewing_cases_to_pass():
    passed = [SimpleNamespace(success=True, best_rank=value) for value in (10, 20, 30)]
    failed = [*passed[:2], SimpleNamespace(success=False, best_rank=101)]

    good = RuntimeProgressiveFitter._readout_quality(passed)
    bad = RuntimeProgressiveFitter._readout_quality(failed)

    assert good.stable
    assert good.pass_at_10 == 1.0
    assert not bad.stable
    assert "below 1.00" in bad.reasons[0]


def stage(name, *, stable):
    return SimpleNamespace(
        name=name,
        lens=SimpleNamespace(metadata={}),
        quality=SimpleNamespace(
            stable=stable,
            reasons=(),
            pass_at_10=0.8 if stable else 0.1,
            rank_overlap=0.9,
        ),
        stage=SimpleNamespace(prompts=8 if name == "Preview" else 32),
        elapsed_seconds=0.1,
    )


class RuntimeDouble:
    model_id = "model/test"

    def __init__(self):
        self.activated = []
        self.metadata = []

    def activate_lens(self, lens, path, quality):
        self.activated.append(quality)
        self.metadata.append(dict(lens.metadata))


class FitterDouble:
    def run(self, *, on_stage, on_progress, cancel_event):
        on_progress("Preview", {"prompt": 1, "prompts": 8})
        on_stage(stage("Preview", stable=False))
        if not cancel_event.is_set():
            on_stage(stage("Stable", stable=True))


class StableFailFitterDouble:
    def run(self, *, on_stage, on_progress, cancel_event):
        on_stage(stage("Preview", stable=False))
        on_stage(
            SimpleNamespace(
                name="Stable",
                lens=SimpleNamespace(metadata={}),
                quality=SimpleNamespace(
                    stable=False,
                    reasons=("rank recovery low",),
                    pass_at_10=0.1,
                    rank_overlap=0.9,
                ),
                stage=SimpleNamespace(prompts=32),
                elapsed_seconds=0.1,
            )
        )


class FitterWithCheckpoint(FitterDouble):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


def test_controller_activates_preview_for_viewing_then_stable(tmp_path):
    runtime = RuntimeDouble()
    controller = ProgressiveLensController(runtime, FitterDouble(), tmp_path)

    controller.start()
    controller.join(timeout=1)

    assert runtime.activated == ["Preview", "Stable"]
    assert runtime.metadata[-1]["quality_gate_version"] == "jspace-v1"
    assert runtime.metadata[-1]["fit_quality_pass_at_10"] == "0.8"
    assert controller.status().state is LensFitState.STABLE


def test_gpu_coordinator_serializes_operations():
    coordinator = GPUCoordinator()
    entered = []
    release = threading.Event()

    def first():
        with coordinator.exclusive("fit"):
            entered.append("fit")
            release.wait(1)

    thread = threading.Thread(target=first)
    thread.start()
    while not entered:
        pass

    def second():
        with coordinator.exclusive("generation"):
            entered.append("generation")

    waiter = threading.Thread(target=second)
    waiter.start()
    assert entered == ["fit"]
    release.set()
    thread.join(1)
    waiter.join(1)
    assert entered == ["fit", "generation"]


def test_controller_fails_closed_when_stable_gate_fails(tmp_path):
    runtime = RuntimeDouble()
    controller = ProgressiveLensController(runtime, StableFailFitterDouble(), tmp_path)

    controller.start()
    controller.join(timeout=1)

    assert runtime.activated == ["Preview"]
    assert controller.status().state is LensFitState.FAILED
    assert "rank recovery low" in controller.status().detail


def test_force_refit_discards_stale_fit_checkpoints(tmp_path):
    fit_dir = tmp_path / ".fit"
    fit_dir.mkdir()
    stale = fit_dir / "stable.fit.pt"
    stale.write_text("bad old checkpoint")
    runtime = RuntimeDouble()
    controller = ProgressiveLensController(
        runtime,
        FitterWithCheckpoint(fit_dir),
        tmp_path,
    )

    controller.start(force=True)
    controller.join(timeout=1)

    assert not stale.exists()
    assert runtime.activated == ["Preview", "Stable"]
