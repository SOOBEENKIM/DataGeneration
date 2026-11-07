"""Full architecture GPU memory/time probe, not performance training."""
import os,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];C=ROOT/'artifacts/meld_compatibility_v1';OUT=ROOT/'artifacts/comparison_completion_v1/meld'
V=C/'vendor/MeLD-b1e5bb0fd95786a7dc11f616892b60ea6b311767'
sys.path[:0]=[str(C/'runtime_deps'),str(V),str(V/'model/LightningD1T')]
os.environ['TORCHDYNAMO_DISABLE']='1'
import torch
from model import timeautoencoder as tae
from models import lightning1dit as dit
from transport import create_transport
from unittest.mock import patch
def main():
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(20261041);tae.device='cuda'
    config=json.loads((OUT/'VAE_CONFIG.json').read_text());data=torch.load(OUT/'DATA.pt',weights_only=False,map_location='cpu')
    rows=[]
    for batch in [1,2]:
      torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();vae=tae.DeapStack(**config).cuda();opt=torch.optim.Adam(vae.parameters(),lr=1e-4)
      ids=data['train'][3][...,1].eq(0).sum(1).argsort(descending=True)[:batch]
      a=[x[ids].cuda() for x in data['train']];start=time.monotonic()
      re,kl=vae.get_loss(*a);(re+.01*kl).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in vae.parameters());opt.step();torch.cuda.synchronize()
      rows.append(dict(stage='VAE',batch=batch,seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,loss=float(re),finite_gradients=True));print(rows[-1],flush=True)
      del vae,opt,a;torch.cuda.empty_cache()
    old=dit.CLIPTextEmbedder
    with patch.object(dit,'CLIPTextEmbedder',lambda hidden_size,dropout_prob=.1:old(hidden_size,model_name=str(C/'clip'),dropout_prob=dropout_prob)):
      model=dit.LightningDiT(in_channels=32,seq_len=config['seq_len'],patch_size=1,hidden_size=256,depth=8,num_heads=8,use_swiglu=True,use_rmsnorm=True,use_checkpoint=True).cuda()
    transport=create_transport('Linear','velocity',use_cosine_loss=True,use_lognorm=True)
    x=torch.randn(2,32,config['seq_len'],device='cuda');torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    loss=transport.training_losses(model,x,dict(y=data['train_text'][:2]))['loss'].mean();loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    rows.append(dict(stage='DiT',batch=2,seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,loss=float(loss),finite_gradients=True));print(rows[-1],flush=True)
    (OUT/'GPU_PREFLIGHT.json').write_text(json.dumps(dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),checks=rows,performance_training=False),indent=2))
if __name__=='__main__':main()
