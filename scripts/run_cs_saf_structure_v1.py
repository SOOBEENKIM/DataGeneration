import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cs_saf_structure_v1 import train,evaluate_saved

p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate'])
p.add_argument('--kappa',type=int,required=True);p.add_argument('--trial',type=int,required=True)
p.add_argument('--model',choices=['U','G','C'],required=True);p.add_argument('--device',default='cuda')
p.add_argument('--smoke-name');a=p.parse_args()
if a.action=='train':train(a.kappa,a.trial,a.model,a.device,a.smoke_name)
else:evaluate_saved(a.kappa,a.trial,a.model)
