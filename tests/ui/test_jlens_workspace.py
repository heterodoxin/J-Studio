from jstudio.ui.lensview.web_view import JLensWebView
from jstudio.ui.lensview.workspace import JLensWorkspace


def test_workspace_uses_original_slice_renderer(qtbot, services):
    workspace = JLensWorkspace(services)
    qtbot.addWidget(workspace)
    workspace.show()

    workspace.inspect("run-1", "( ^ )", "ASCII face", position=1)

    qtbot.waitUntil(lambda: workspace.web.current_generation == 1)
    assert isinstance(workspace.web, JLensWebView)
    assert workspace.status.text() == "Ready"
    assert workspace.selection.position == 1
    assert not hasattr(workspace, "matrix")


def test_refresh_requests_a_new_slice_generation(qtbot, services):
    workspace = JLensWorkspace(services)
    qtbot.addWidget(workspace)
    workspace.inspect("run-1", "ascii", "ASCII")
    qtbot.waitUntil(lambda: workspace.web.current_generation == 1)

    workspace.refresh_button.click()

    qtbot.waitUntil(lambda: workspace.web.current_generation == 2)


def test_web_selection_updates_shared_selection(qtbot, services):
    workspace = JLensWorkspace(services)
    qtbot.addWidget(workspace)
    workspace.inspect("run-1", "ascii", "ASCII")

    workspace.web.bridge.select(28, 42)

    assert workspace.selection.position == 28
    assert workspace.selection.layer == 42


def test_workspace_tracks_streamed_run_text_without_expensive_refresh(qtbot, services):
    workspace = JLensWorkspace(services)
    qtbot.addWidget(workspace)
    workspace.inspect("run-1", "prompt", "Live")
    qtbot.waitUntil(lambda: workspace.web.current_generation == 1)

    workspace.stream_text("run-1", "prompt\n\nfirst streamed token")

    assert workspace._request.text == "prompt\n\nfirst streamed token"
    assert workspace.web.current_generation == 1
    assert "streaming" in workspace.status.text().lower()


def test_long_slice_requests_default_to_recent_token_window(qtbot, services):
    workspace = JLensWorkspace(services)
    qtbot.addWidget(workspace)
    text = " ".join(f"tok{i}" for i in range(300))

    workspace.inspect("run-long", text, "Long")

    assert workspace._request.last_n_tokens == 256


def test_lens_badge_flags_sketched_readout_as_unreliable(qtbot, services):
    workspace = JLensWorkspace(services)
    qtbot.addWidget(workspace)

    workspace.set_lens_identity("sketched-jacobian-r128 · pass@10 0.31")
    assert workspace.lens_badge.text().startswith("⚠")
    assert "unreliable" in workspace.lens_badge.toolTip()

    workspace.set_lens_identity("dense-jacobian-n16 · pass@10 0.62")
    assert workspace.lens_badge.text() == "dense-jacobian-n16 · pass@10 0.62"
    assert "unreliable" not in workspace.lens_badge.toolTip()

    workspace.set_lens_identity(None)
    assert workspace.lens_badge.text() == "No lens"
