import importlib.util
from pathlib import Path


def test_idle_mps_daemon_does_not_hide_active_or_unknown_workloads():
    path=Path(__file__).resolve().parents[1]/'scripts/dispatch_argn_label_first_control.py'
    spec=importlib.util.spec_from_file_location('order_dispatch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows='\n'.join([
        'gpu0, 1, nvidia-cuda-mps-server, 28',
        'gpu1, 1, /usr/bin/nvidia-cuda-mps-server, 28',
        'gpu1, 2, python, 100',
        'gpu2, 1, nvidia-cuda-mps-server, 500',
        'gpu3, 1, nvidia-cuda-mps-server, [N/A]',
        'gpu4, 2, python, 1',
    ])
    assert module.active_workload_gpus(rows)=={'gpu1','gpu2','gpu3','gpu4'}
