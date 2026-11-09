import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_hust_vspec import run_with_backend


def _selected_backend(monkeypatch, tmp_path):
    vendor = tmp_path / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    library = vendor / "op_api/lib/libcust_opapi.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"test")
    module = tmp_path / "vllm_ascend/__init__.py"
    module.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        run_with_backend.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(module)),
    )
    return vendor


def test_launch_uses_selected_installed_library(monkeypatch, tmp_path):
    vendor = _selected_backend(monkeypatch, tmp_path)
    observed = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-with-backend", "--backend-package", str(tmp_path), "--", sys.executable, "-V"],
    )
    monkeypatch.setattr(
        run_with_backend.os,
        "execvpe",
        lambda executable, command, env: observed.update(
            executable=executable, command=command, env=env
        ),
    )
    run_with_backend.main()
    assert observed["executable"] == sys.executable
    assert observed["command"] == [sys.executable, "-V"]
    assert observed["env"]["ASCEND_CUSTOM_OPP_PATH"] == str(vendor)


def test_launch_rejects_different_backend_import(monkeypatch, tmp_path):
    _selected_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_with_backend.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin="/another/env/vllm_ascend/__init__.py"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-with-backend", "--backend-package", str(tmp_path), "--", sys.executable, "-V"],
    )
    with pytest.raises(RuntimeError, match="does not import the selected"):
        run_with_backend.main()


def test_launch_rejects_command_from_other_python(monkeypatch, tmp_path):
    _selected_backend(monkeypatch, tmp_path)
    foreign = tmp_path / "foreign" / "bin" / "worker"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/another/env/bin/python\nprint(1)\n", encoding="utf-8")
    foreign.chmod(0o755)
    monkeypatch.setattr(
        sys, "argv", ["run-with-backend", "--backend-package", str(tmp_path), "--", str(foreign)]
    )
    with pytest.raises(RuntimeError, match="different Python environment"):
        run_with_backend.main()


def test_launch_accepts_console_script_from_same_python(monkeypatch, tmp_path):
    _selected_backend(monkeypatch, tmp_path)
    script = tmp_path / "worker"
    script.write_text(f"#!{sys.executable}\nprint(1)\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(
        sys, "argv", ["run-with-backend", "--backend-package", str(tmp_path), "--", str(script)]
    )
    observed = {}
    monkeypatch.setattr(
        run_with_backend.os,
        "execvpe",
        lambda executable, command, env: observed.update(
            executable=executable, command=command, env=env
        ),
    )
    run_with_backend.main()
    assert Path(observed["executable"]) == script
