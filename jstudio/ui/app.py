"""J Studio desktop application entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from jstudio.domain import SessionState
from jstudio.project import ProjectDocument
from jstudio.services.fake import create_fake_services
from jstudio.services.hf_runtime import DEFAULT_MODEL_ID, create_hf_services
from jstudio.services.protocols import JStudioServices
from jstudio.ui.shell.main_window import JStudioMainWindow
from jstudio.ui.startup import choose_startup_model
from jstudio.ui.theme import apply_jstudio_theme


def create_application(
    argv: Sequence[str] | None = None,
    *,
    services: JStudioServices | None = None,
    project: ProjectDocument | None = None,
    application: QApplication | None = None,
) -> tuple[QApplication, JStudioMainWindow]:
    """Build an application and ready window without entering the event loop."""
    app = application or QApplication.instance() or QApplication(list(argv or ()))
    app.setOrganizationName("J Studio")
    app.setApplicationName("J Studio")
    app.setApplicationVersion("0.1.0")
    apply_jstudio_theme(app)
    icon = QIcon(str(Path(__file__).parents[1] / "assets" / "j-studio.png"))
    app.setWindowIcon(icon)
    selected_services = services or create_fake_services()
    window = JStudioMainWindow(selected_services, project or ProjectDocument.new())
    window.setWindowIcon(icon)
    sessions = selected_services.sessions.list_sessions()
    session = next(
        (candidate for candidate in sessions if candidate.state is SessionState.READY),
        sessions[0] if sessions else None,
    )
    window.set_session(session)
    return app, window


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J Studio model research workbench")
    parser.add_argument("project", nargs="?", type=Path)
    parser.add_argument("--screenshot", type=Path, help="save the initial window as PNG")
    parser.add_argument("--quit-after", type=int, metavar="MS")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use deterministic demo data instead of a local model",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face decoder model ID or local path",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face to download a model missing from the local cache",
    )
    parser.add_argument(
        "--lens",
        type=Path,
        help="path to a fitted Jacobian lens (defaults to the model lens cache)",
    )
    return parser.parse_args(argv)


def select_services(args: argparse.Namespace) -> JStudioServices:
    if args.demo:
        return create_fake_services()
    return create_hf_services(
        args.model or DEFAULT_MODEL_ID,
        local_files_only=not args.allow_download,
        lens_path=args.lens,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    app = QApplication.instance() or QApplication(list(argv or ()))
    if not args.demo and args.model is None:
        selection = choose_startup_model()
        if selection is None:
            return 1
        args.model, selected_lens = selection
        if selected_lens is not None:
            args.lens = selected_lens
    try:
        services = select_services(args)
    except ValueError as exc:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, "J Studio", f"Could not load the selected lens:\n{exc}")
        return 1
    app, window = create_application(argv, services=services, application=app)
    if args.project:
        window.open_project(args.project)
    window.show()
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            window.grab().save(str(args.screenshot), "PNG")

        QTimer.singleShot(150, capture)
    if args.quit_after is not None:
        QTimer.singleShot(max(0, args.quit_after), app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
