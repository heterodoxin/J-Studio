import pytest

from jstudio.project import ProjectDocument
from jstudio.services.fake import create_fake_services
from jstudio.ui.shell.main_window import JStudioMainWindow


@pytest.fixture
def services():
    result = create_fake_services(token_delay=0.001)
    yield result
    result.generation.close()


@pytest.fixture
def project():
    return ProjectDocument.new("Test Project")


@pytest.fixture
def window(qtbot, services, project):
    result = JStudioMainWindow(services, project)
    qtbot.addWidget(result)
    result.set_session(services.sessions.list_sessions()[0])
    result.show()
    return result
