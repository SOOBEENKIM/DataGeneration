"""Pinned official latent diffusion/CFG execution smoke; not adequate training."""
import json,os,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/meld_compatibility_v1'
V=OUT/'vendor/MeLD-b1e5bb0fd95786a7dc11f616892b60ea6b311767'
sys.path[:0]=[str(OUT/'runtime_deps'),str(V),str(V/'model/LightningD1T')]
from models import lightning1dit as dit
from transport import create_transport,Sampler
from model import timeautoencoder as tae

def write(name,v):(OUT/'report'/name).write_text(json.dumps(v,indent=2)+'\n')
def main():
 torch.manual_seed(90210);torch.set_num_threads(2);tae.device='cpu';start=time.monotonic()
 old=dit.CLIPTextEmbedder
 def public_clip(hidden_size,dropout_prob=.1):return old(hidden_size,model_name=str(OUT/'clip'),dropout_prob=dropout_prob)
 with patch.object(dit,'CLIPTextEmbedder',public_clip):
  model=dit.LightningDiT(in_channels=8,seq_len=8,patch_size=1,hidden_size=64,depth=2,num_heads=4)
 payload=torch.load(OUT/'toy_v3/LATENT_SMOKE.pt',weights_only=True,map_location='cpu');x=payload['latent'].permute(0,2,1)
 # Explicit toy static conditions contain no fraud labels, future gaps or lengths.
 conditions=['gender F; state CA; birth year 1970; city population 100000','gender M; state NY; birth year 1980; city population 500000']
 model.eval()
 with torch.no_grad():
  empty=model.y_embedder(['',None],False);torch.testing.assert_close(empty[0],empty[1],rtol=0,atol=0)
  given=model.y_embedder(conditions,False);assert torch.isfinite(given).all() and not torch.equal(given[0],given[1])
  dropped=model.y_embedder(conditions,False,force_drop_ids=torch.ones(2,dtype=torch.long));torch.testing.assert_close(dropped,empty,rtol=0,atol=0)
 transport=create_transport('Linear','velocity',use_cosine_loss=False,use_lognorm=False)
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4);curve=[]
 for step in range(4):
  model.train();losses=transport.training_losses(model,x,dict(y=conditions));loss=losses['loss'].mean();assert torch.isfinite(loss)
  optimizer.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step();curve.append(float(loss));print('DIT_STEP',step,float(loss),flush=True)
 model.eval();sampler=Sampler(transport).sample_ode(sampling_method='euler',num_steps=4)
 with torch.no_grad():
  generated=sampler(torch.randn_like(x),model,y=conditions)[-1]
  cfg=model.forward_with_cfg(torch.randn(4,8,8),torch.ones(4)*.5,conditions+['',''],cfg_scale=1.5)
  assert generated.shape==x.shape and torch.isfinite(generated).all() and torch.isfinite(cfg).all()
 info=json.loads((OUT/'toy_v3/ENCODE_DECODE.json').read_text())['datatype'];ck=torch.load(OUT/'toy_v3/smoke_vae.pt',weights_only=True,map_location='cpu')
 vae=tae.DeapStack(channels=0,batch_size=2,seq_len=8,n_bins=info['n_bins'],n_cats=info['n_cats'],n_nums=info['n_nums'],cards=info['cards'],feature_size=sum(info[k] for k in ['n_bins','n_cats','n_nums']),hidden_size=64,num_layers=2,bidirectional=False,emb_dim=32,time_dim=8,lat_dim=8)
 vae.load_state_dict(ck['state_dict']);vae.eval()
 with torch.no_grad():decoded=vae.decoder(generated.permute(0,2,1))
 assert decoded['eos'].shape==(2,8,2)
 torch.save(dict(latent=generated,decoded=decoded),OUT/'TOY_DIFFUSION_SMOKE.pt')
 # The code accepts the full required length; patch-size padding is explicit, not truncation.
 maximum=json.loads((OUT/'sparkov_v3/ENCODE_DECODE.json').read_text())['max_length'];padded=maximum+(maximum%2)
 with patch.object(dit,'CLIPTextEmbedder',public_clip):long=dit.LightningDiT(in_channels=8,seq_len=padded,patch_size=2,hidden_size=64,depth=2,num_heads=4)
 with torch.no_grad():q=long(torch.zeros(1,8,padded),torch.tensor([.5]),y=[''])
 assert q.shape==(1,8,padded) and torch.isfinite(q).all()
 write('DIFFUSION_SMOKE_COMPLETE.json',dict(steps=4,loss_curve=curve,ode_steps=4,generated_shape=list(generated.shape),max_real_length=maximum,explicit_padding=padded-maximum,full_length_forward=True,text_condition_and_null_checks=True,seconds=time.monotonic()-start,architecture='tiny64hidden/2layer/4head; not paper256hidden/8layer/8head',public_clip_adapter=True,adequate_training=False,full_sparkov_generation=False,test_reads=0))
 print('DIT_SMOKE_COMPLETE',flush=True)
if __name__=='__main__':main()
