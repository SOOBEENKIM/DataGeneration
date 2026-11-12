"""Recorded test-fixture precision amendment; model, data and fits are unchanged."""
import argparse
import json
from pathlib import Path
import run_argn_clock_regression as study
from run_argn_time_density import ROOT,digest
from diagnose_argn_residual import verify_registry


def verify():
    verify_registry()
    a=json.loads((study.OUT/'AMENDMENT_01.json').read_text())
    assert digest(a['archived_original_test'])==a['original_test_sha256']
    assert digest(__file__)==a['launcher_sha256']
    assert digest(study.OUT/'time_head.pt')==a['unchanged_head_sha256']
    for file in [ROOT/'artifacts/argn_time_density_v1/MANIFEST.json',study.OUT/'MANIFEST.json']:
        m=json.loads(file.read_text())
        for p,h in m['hashes'].items():
            if p==a['test_path']:
                assert h==a['original_test_sha256'];h=a['corrected_test_sha256']
            assert digest(p)==h,p


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','dispatch']);p.add_argument('--seed',type=int,choices=study.CFG['fit_seeds']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    study.verify=verify;study.verify_for_dispatch=verify;study.__file__=str(Path(__file__).resolve())
    study.worker(a.seed) if a.mode=='worker' else study.dispatch()
