from PySide6.QtCore import QUrl

from jstudio.services.protocols import SlicePage
from jstudio.ui.lensview.web_view import JLensBridge, JLensWebView, SecureSlicePage


def test_web_view_loads_html_and_rejects_remote_navigation(qtbot):
    view = JLensWebView()
    qtbot.addWidget(view)

    view.set_page(SlicePage("r", 1, "<html><body>slice</body></html>"))

    assert view.current_generation == 1
    assert view.last_html.endswith("</html>")
    assert view.url().isLocalFile()
    assert view.page().allows(view.url())
    assert not view.page().allows(QUrl("https://example.com"))
    assert not view.page().allows(QUrl("file:///tmp/untrusted.html"))


def test_bridge_validates_payloads(qtbot):
    bridge = JLensBridge()

    with qtbot.assertNotEmitted(bridge.intervention_requested):
        bridge.intervene("", -1, -1)

    with qtbot.waitSignal(bridge.intervention_requested) as signal:
        bridge.intervene("nose", 42, 28)
    assert signal.args == ["nose", 42, 28]


def test_secure_page_type_is_used(qtbot):
    view = JLensWebView()
    qtbot.addWidget(view)
    assert isinstance(view.page(), SecureSlicePage)
