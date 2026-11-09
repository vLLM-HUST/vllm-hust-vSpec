import sys

from vllm_hust_vspec import run_with_backend


def test_launch_uses_selected_installed_library(monkeypatch, tmp_path):
    vendor = tmp_path / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
    library = vendor / 'op_api/lib/libcust_opapi.so'
    library.parent.mkdir(parents=True)
    library.write_bytes(b'test')
    observed = {}
    monkeypatch.setattr(sys, 'argv', ['run-with-backend', '--backend-package', str(tmp_path), '--', 'python', '-V'])
    monkeypatch.setattr(run_with_backend.os, 'execvpe', lambda executable, command, env:
                        observed.update(executable=executable, command=command, env=env))
    run_with_backend.main()
    assert observed['executable'] == 'python'
    assert observed['command'] == ['python', '-V']
    assert observed['env']['ASCEND_CUSTOM_OPP_PATH'] == str(vendor)
