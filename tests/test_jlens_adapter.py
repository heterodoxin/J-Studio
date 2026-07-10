import inspect

import pytest

import jstudio.services.jlens_adapter as module
from jstudio.services.jlens_adapter import JLensAdapter, JLENSUnavailableError


def test_adapter_reports_precise_unavailable_capability():
    adapter = JLensAdapter(module_name="module_that_does_not_exist")

    assert not adapter.available
    assert "module_that_does_not_exist" in adapter.unavailable_reason
    with pytest.raises(JLENSUnavailableError):
        adapter.require_available()


def test_adapter_never_imports_qt():
    source = inspect.getsource(module)
    assert "PySide" not in source
    assert "Qt" not in source
