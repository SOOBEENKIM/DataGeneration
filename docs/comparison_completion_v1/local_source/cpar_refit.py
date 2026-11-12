"""Run the exact frozen CPAR adequacy routine with one new independent seed."""
import sys, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
src=ROOT/'artifacts/cpar_adequacy_v1/local_source/study_cpar.py'
spec=importlib.util.spec_from_file_location('frozen_cpar_adequacy',src)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_verify=m.verify
seed=int(sys.argv[1]);assert seed in [20261022,20261023]
newout=ROOT/f'artifacts/comparison_completion_v1/cpar/{seed}'
oldout=m.OUT
def verify():
    m.OUT=oldout
    try:original_verify()
    finally:m.OUT=newout
    manifest=m.json.loads((newout.parent.parent/'MANIFEST.json').read_text())
    for p,h in manifest['hashes'].items():assert m.digest(p)==h,p
m.verify=verify;m.OUT=newout;m.SEED=seed
if __name__=='__main__':m.main()
