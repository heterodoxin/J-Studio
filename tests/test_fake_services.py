import threading

from jstudio.domain import LensFitState, RunMode, RunState
from jstudio.services.fake import create_fake_services
from jstudio.services.protocols import GenerationRequest, SliceRequest


class Sink:
    def __init__(self):
        self.started = None
        self.tokens = []
        self.frames = []
        self.finished = None
        self.done = threading.Event()

    def on_started(self, run):
        self.started = run

    def on_token(self, run_id, token, output_text):
        self.tokens.append((token, output_text))

    def on_frame(self, frame):
        self.frames.append(frame)

    def on_intervention(self, intervention_id, state, detail):
        pass

    def on_finished(self, run):
        self.finished = run
        self.done.set()

    def on_error(self, run_id, message, detail=""):
        raise AssertionError((run_id, message, detail))


def test_fake_generation_streams_tokens_and_frames():
    services = create_fake_services(token_delay=0.001)
    session = services.sessions.list_sessions()[0]
    sink = Sink()

    run_id = services.generation.start(
        GenerationRequest(session.session_id, "Inspect this", RunMode.BASELINE), sink
    )

    assert sink.done.wait(2)
    assert sink.started.run_id == run_id
    assert sink.finished.state is RunState.COMPLETE
    assert sink.tokens and sink.frames
    assert services.lens.frames(run_id) == tuple(sink.frames)
    services.generation.close()


def test_fake_generation_supports_pause_next_and_stop():
    services = create_fake_services(token_delay=0.05)
    session = services.sessions.list_sessions()[0]
    sink = Sink()
    run_id = services.generation.start(
        GenerationRequest(session.session_id, "Inspect this"), sink
    )

    services.generation.pause(run_id)
    services.generation.next_token(run_id)
    services.generation.stop(run_id)

    assert sink.done.wait(2)
    assert sink.finished.state is RunState.CANCELLED
    services.generation.close()


def test_fake_lens_service_returns_self_contained_slice_page():
    services = create_fake_services(token_delay=0)

    page = services.lens.request_slice(
        SliceRequest("run-1", "( ^ )", "ASCII face")
    ).result(timeout=1)

    assert page.run_id == "run-1"
    assert page.generation == 1
    assert "J-lens" in page.html
    assert "ASCII face" in page.html
    services.generation.close()


def test_fake_lens_fit_publishes_preview_then_stable():
    services = create_fake_services(token_delay=0)
    seen = []
    unsubscribe = services.lens.subscribe_fit(seen.append)

    services.lens.start_fit()

    assert [status.state for status in seen] == [
        LensFitState.PREVIEW,
        LensFitState.REFINING,
        LensFitState.STABLE,
    ]
    assert services.lens.fit_status().quality == "passed"
    unsubscribe()
