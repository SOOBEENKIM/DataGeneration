"""Official LBDTPP core, explicit Sparkov adapter, no automatic test loader."""
import argparse, copy, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/lbdtpp_sparkov_v1'
SOURCE=ROOT/'artifacts/reference_sources/lbdtpp_20260927'
sys.path[:0]=[str(OUT/'vendor'),str(SOURCE)]
from model.bdtpp_model import BDTPPModel
import model.diffusion_model as diffusion_module
from tqdm.auto import tqdm
diffusion_module.tqdm=lambda *a,**kw:tqdm(*a,**{**kw,'disable':True})

CFG=dict(seed=20260929,epochs=50,batch_size=4,accumulation=8,lr=.0005,
         block_size=8,transformer_dim=32,transformer_heads=4,num_decoder_layers=4,
         dropout=0.,diffusion_steps=100,sampling_steps=50,loss_lambda=1.,
         generation_seeds=[20261011,20261012],parent_seed=20260930)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert manifest['config']==CFG
    for p,h in manifest['hashes'].items():assert digest(p)==h,p

def raw(split):
    return pd.read_parquet(ROOT/f'artifacts/argn_residual_v1/metadata_{split}.parquet')

def load_data(categories,scale,split):
    result=[]
    for eid,g in raw(split).groupby('entity_id',sort=False):
        g=g.sort_values('event_index');assert np.array_equal(g.event_index,np.arange(len(g)))
        marks=2*g.category.map(categories).to_numpy()+g.label.to_numpy()
        assert pd.notna(marks).all()
        result.append((str(eid),torch.tensor(g.gap.fillna(0).to_numpy()/scale,dtype=torch.float32),
                       torch.tensor(marks,dtype=torch.long)))
    return result

def batches(data,shuffle,rng):
    # Full sequences; no 1024-event or sliding-window truncation. Nearby lengths
    # are grouped to bound padding memory. Shuffle groups each epoch.
    indices=np.argsort([len(x[1]) for x in data],kind='stable')
    groups=[indices[i:i+CFG['batch_size']] for i in range(0,len(indices),CFG['batch_size'])]
    if shuffle:rng.shuffle(groups)
    return groups

def collate(data,indices,classes):
    length=int(math.ceil(max(len(data[i][1]) for i in indices)/8)*8)
    x=torch.zeros(len(indices),length,device='cuda')
    e=torch.full((len(indices),length),classes,dtype=torch.long,device='cuda')
    for j,i in enumerate(indices):
        n=len(data[i][1]);x[j,:n]=data[i][1].to('cuda');e[j,:n]=data[i][2].to('cuda')
    return x,e

@torch.no_grad()
def validation(model,data,classes):
    model.eval();sums=np.zeros(4);count=0
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261321);torch.cuda.manual_seed_all(20261321)
        for ix in batches(data,False,None):
            vals=model(*collate(data,ix,classes),task='unconditional')
            sums+=np.array([v.item() for v in vals])*len(ix);count+=len(ix)
    return (sums/count).tolist()

def fit_and_generate(normalization):
    verify();assert torch.cuda.is_available()
    run=OUT/normalization;run.mkdir(exist_ok=False)
    torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed']);random.seed(CFG['seed'])
    m=raw('optimization');categories={x:i for i,x in enumerate(sorted(m.category.unique()))}
    scale=float(m.groupby('entity_id').gap.sum().max()) if normalization=='official_scale' else float(m.loc[m.gap.gt(0),'gap'].median())
    data=load_data(categories,scale,'optimization');val=load_data(categories,scale,'internal_validation')
    classes=len(categories)*2
    args=SimpleNamespace(**{k:v for k,v in CFG.items() if k in ['batch_size','block_size','transformer_dim',
         'transformer_heads','num_decoder_layers','dropout','diffusion_steps','sampling_steps','loss_lambda']},
         num_classes=classes,model_length=int(math.ceil(max(len(x[1]) for x in data)/8)*8),tgt_len=20,device='cuda:0')
    model=BDTPPModel(args,classes).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=CFG['lr'])
    groups=batches(data,False,None);updates=math.ceil(len(groups)/CFG['accumulation'])*CFG['epochs']
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates,eta_min=0.)
    # Preserve the released EMA(0.9) convention; here update follows optimizer.
    ema={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
    best=float('inf');curve=[];started=time.monotonic();rng=np.random.default_rng(CFG['seed'])
    write(run/'START.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),config=vars(args),
          normalization=normalization,scale=scale,categories=categories,train_rows=len(m),train_customers=len(data),
          first_event_retained=True,full_length=True,torch_version=torch.__version__,started_utc=datetime.now(timezone.utc).isoformat()))
    for epoch in range(1,CFG['epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True);sums=np.zeros(4);count=0
        groups=batches(data,True,rng)
        for start in range(0,len(groups),CFG['accumulation']):
            block=groups[start:start+CFG['accumulation']];denom=sum(len(ix) for ix in block)
            for ix in block:
                values=model(*collate(data,ix,classes),task='unconditional')
                assert all(torch.isfinite(v) for v in values)
                (values[0]*len(ix)/denom).backward()
                sums+=np.array([v.item() for v in values])*len(ix);count+=len(ix)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step()
            with torch.no_grad():
                for k,p in model.named_parameters():
                    if p.requires_grad:ema[k].lerp_(p,.1)
        row=dict(epoch=epoch,seconds=time.monotonic()-started,train=(sums/count).tolist())
        if epoch%2==0 or epoch==1:
            original={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
            with torch.no_grad():
                for k,p in model.named_parameters():
                    if p.requires_grad:p.copy_(ema[k])
            values=validation(model,val,classes);row['internal']=values
            if values[0]<best:
                best=values[0];torch.save(dict(state_dict=model.state_dict(),args=vars(args),scale=scale,
                    categories=categories,epoch=epoch,internal=values),run/'best.pt')
            with torch.no_grad():
                for k,p in model.named_parameters():
                    if p.requires_grad:p.copy_(original[k])
        curve.append(row);write(run/'LEARNING_CURVE.json',curve)
        print('LBD_EPOCH',normalization,row,flush=True)
    checkpoint=torch.load(run/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    write(run/'FIT_COMPLETE.json',dict(selected_epoch=checkpoint['epoch'],internal=checkpoint['internal'],
          seconds=time.monotonic()-started,sha256=digest(run/'best.pt'),independent_fits=1,test_events_read=False))
    categories_reverse={v:k for k,v in categories.items()}
    for gs in CFG['generation_seeds']:
        torch.manual_seed(gs);torch.cuda.manual_seed_all(gs)
        schedule_path=ROOT/f'artifacts/argn_clock_initial_coupling_v1/runs/init_only_20260930/generated_validation_{gs}.parquet'
        # Common externally supplied lengths come from the frozen generator,
        # never from target customer event data or their final timestamps.
        schedule=pd.read_parquet(schedule_path,columns=['entity_id']).groupby('entity_id',sort=False).size()
        rows=[];generation_started=time.monotonic()
        for start in range(0,len(schedule),16):
            cohort=schedule.iloc[start:start+16];bs=len(cohort)
            model.block_dit.batch_size=bs;model.block_dit.reset_kv_cache()
            xs=[];es=[]
            with torch.inference_mode():
                for block in range(int(math.ceil(cohort.max()/8))):
                    latent=model.diffusion_model.uncond_sample(batch_size=bs)
                    xs.append(model.time_decoder(latent).squeeze(-1).cpu().numpy()*scale)
                    es.append(model.type_decoder(latent).argmax(-1).cpu().numpy())
                x=np.concatenate(xs,1);e=np.concatenate(es,1)
            for j,(eid,n) in enumerate(cohort.items()):
                for t in range(int(n)):
                    mark=int(e[j,t]);rows.append(dict(entity_id=eid,event_index=t,
                        gap=float(x[j,t]) if t else np.nan,event_is_fraud=str(mark%2),
                        category=categories_reverse[mark//2]))
            print('LBD_GENERATE',normalization,gs,start+bs,len(schedule),flush=True)
            model.block_dit.remove_kv_cache()
        d=pd.DataFrame(rows);path=run/f'generated_validation_{gs}.parquet';d.to_parquet(path,index=False)
        write(run/f'GENERATION_{gs}.json',dict(rows=len(d),seconds=time.monotonic()-generation_started,
            shared_length_schedule=str(schedule_path),schedule_sha256=digest(schedule_path),
            sha256=digest(path),oracle_timestamps_used=False,customer_context_used=False,
            note='count-conditioned timing/label/category comparator, not full mixed-field generation'))
    verify();write(run/'COMPLETE.json',dict(fits=1,generations=2,test_events_read=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('normalization',choices=['official_scale','median_scale']);a=p.parse_args()
    fit_and_generate(a.normalization)
