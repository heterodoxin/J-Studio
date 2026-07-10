from jstudio.domain import LensFitState


def test_lens_fit_status_transitions_capabilities(qtbot, window, services):
    services.lens.publish_fit(LensFitState.PREVIEW, 8, 8, "unchecked")
    qtbot.waitUntil(lambda: "Preview" in window.session_bar.lens_status.text())

    assert window.current_session.capabilities.inspect
    assert not window.current_session.capabilities.intervene
    assert window.jlens_workspace.web.isEnabled()

    services.lens.publish_fit(LensFitState.STABLE, 32, 32, "passed")
    qtbot.waitUntil(lambda: "Stable" in window.session_bar.lens_status.text())

    assert window.current_session.capabilities.inspect
    assert window.current_session.capabilities.intervene
    assert window.jlens_workspace.web.isEnabled()
