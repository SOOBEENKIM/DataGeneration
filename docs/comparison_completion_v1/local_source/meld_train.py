"""Full pinned-public-code MeLD adaptation. Never imports or reads sealed test."""
import os,sys,json,pickle,random,time,copy,hashlib
from pathlib import Path
from unittest.mock import patch
os.environ['TORCHDYNAMO_DISABLE']='1'
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/comparison_completion_v1';DATA=OUT/'meld';COMPAT=ROOT/'artifacts/meld_compatibility_v1'
V=COMPAT/'vendor/MeLD-b1e5bb0fd95786a7dc11f616892b60ea6b311767'
sys.path[:0]=[str(COMPAT/'runtime_deps'),str(V),str(V/'model/LightningD1T')]
import numpy as np
import pandas as pd
import torch
from model import timeautoencoder as tae
from models import lightning1dit as dit
from transport import create_transport,Sampler
from transformers import CLIPTextModel
def write(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');q.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def setseed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def rng():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore(s):torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def verify():
    for p,h in json.loads((OUT/'MELD_MANIFEST.json').read_text())['hashes'].items():assert digest(p)==h,p
def make_dit(L):
    old=dit.CLIPTextEmbedder
    with patch.object(dit,'CLIPTextEmbedder',lambda hidden_size,dropout_prob=.1:old(hidden_size,model_name=str(COMPAT/'clip'),dropout_prob=dropout_prob)):
        m=dit.LightningDiT(in_channels=32,seq_len=L,patch_size=1,hidden_size=256,depth=8,num_heads=8,use_swiglu=True,use_rmsnorm=True,use_checkpoint=True)
    reference=CLIPTextModel.from_pretrained(str(COMPAT/'clip'));state=reference.state_dict()
    changed=[k for k,v in m.y_embedder.text_encoder.state_dict().items() if not torch.equal(v,state[k])]
    m.y_embedder.text_encoder.load_state_dict(state)
    for k,v in m.y_embedder.text_encoder.state_dict().items():assert torch.equal(v,state[k])
    for p in m.y_embedder.text_encoder.parameters():p.requires_grad_(False)
    m.y_embedder.text_encoder.eval();del reference
    return m.cuda(),changed
@torch.no_grad()
def vae_eval(m,a,seed):
    m.eval();n=len(a[0]);valid_total=float(a[3][...,1].eq(0).sum());re_sum=kl_sum=0.;conf=np.zeros((2,2),dtype=np.int64);bce_sum=0.;eos_correct=eos_total=0
    with torch.random.fork_rng(devices=[0]):
      torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
      for start in range(0,n,8):
        x,t,miss,mask=[v[start:start+8].cuda() for v in a];out,z,mu,lv=m(x,t,miss,mask)
        disc,num=tae.auto_loss(x,t,miss,mask,out,m.n_bins,m.n_nums,m.n_cats,m.cards)
        valid=mask[...,1].eq(0);v=float(valid.sum());re_sum+=float(disc+num)*v;kl_sum+=float(-.5*(1+lv-mu.pow(2)-lv.exp()).mean())*len(x)
        truth=x[...,0].long();pred=out['bins'][...,0].gt(0).long()
        for i in [0,1]:
          for j in [0,1]:conf[i,j]+=int(((truth==i)&(pred==j)&valid).sum())
        bce_sum+=float(torch.nn.functional.binary_cross_entropy_with_logits(out['bins'][...,0],truth.float(),reduction='none')[valid].sum())
        eos=out['eos'].argmax(-1);eos_correct+=int(((eos==mask[...,0])&valid).sum());eos_total+=int(valid.sum())
    return dict(reconstruction=re_sum/valid_total,kl=kl_sum/n,confusion=conf.tolist(),fraud_recall=float(conf[1,1]/max(conf[1].sum(),1)),fraud_precision=float(conf[1,1]/max(conf[:,1].sum(),1)),fraud_bce=bce_sum/valid_total,eos_accuracy=eos_correct/eos_total)
def train_vae(config,data,run,fs):
    m=tae.DeapStack(**config).cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-6)
    best=float('inf');best_epoch=0;cap=.01;beta_wait=0;start_epoch=1;elapsed_start=time.monotonic()
    if (run/'vae_last.pt').exists():
        s=torch.load(run/'vae_last.pt',map_location='cpu',weights_only=False);m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);restore(s['rng']);start_epoch=s['epoch']+1;best=s['best'];best_epoch=s['best_epoch'];cap=s['cap'];beta_wait=s['beta_wait']
    train=data['train'];N=len(train[0]);curve=[]
    for epoch in range(start_epoch,1001):
        m.train();order=torch.randperm(N);beta=.01 if epoch<=50 else cap*min(1.,(epoch%5)/4);total=0.
        for j in range(0,N,128):
            idx=order[j:j+128];den=float(train[3][idx,...,1].eq(0).sum());opt.zero_grad(set_to_none=True)
            for k in range(0,len(idx),8):
                part=idx[k:k+8];a=[x[part].cuda() for x in train];re,kl=m.get_loss(*a)
                loss=re*(float(a[3][...,1].eq(0).sum())/den)+beta*kl*(len(part)/len(idx));assert torch.isfinite(loss)
                loss.backward();total+=float(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
        rec=dict(epoch=epoch,beta=beta,train_macro_loss=total/((N+127)//128),seconds=time.monotonic()-elapsed_start)
        if epoch==1 or epoch%5==0:
            validation=vae_eval(m,data['internal'],fs+900);rec.update(internal=validation)
            if validation['reconstruction']<best:
                best=validation['reconstruction'];best_epoch=epoch;beta_wait=0;save(run/'vae_best.pt',dict(model=m.state_dict(),epoch=epoch,internal=validation,config=config))
            else:beta_wait+=5
            if epoch>50 and beta_wait>=20 and cap>1e-5:cap=max(1e-5,cap/2);beta_wait=0
        curve.append(rec)
        with (run/'VAE_CURVE.jsonl').open('a') as h:h.write(json.dumps(rec)+'\n')
        write(run/'STATUS.json',dict(stage='VAE_training',epoch=epoch,max_epochs=1000,best_epoch=best_epoch,best_internal_reconstruction=best,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),seconds=rec['seconds']))
        save(run/'vae_last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,cap=cap,beta_wait=beta_wait,rng=rng()))
        print('VAE_EPOCH',fs,epoch,best_epoch,best,rec['seconds'],flush=True)
        if epoch>=50 and epoch-best_epoch>=100:break
    s=torch.load(run/'vae_best.pt',map_location='cpu',weights_only=False);m.load_state_dict(s['model']);m.eval();check=vae_eval(m,data['internal'],fs+900)
    write(run/'VAE_COMPLETE.json',dict(last_epoch=epoch,selected_epoch=s['epoch'],internal=check,parameters=sum(p.numel() for p in m.parameters()),test_reads=0))
    if check['confusion'][1][1]==0:
        write(run/'ADEQUACY_BLOCK.json',dict(stage='VAE',reason='zero reconstructed true-positive internal fraud events',internal=check));return None
    return m
@torch.no_grad()
def encode(m,arrays):
    z=[]
    for start in range(0,len(arrays[0]),8):z.append(m.encoder(*[x[start:start+8].cuda() for x in arrays])[0].cpu())
    return torch.cat(z)
def losses(transport,m,x,text):
    out=transport.training_losses(m,x,dict(y=text));return out['loss'].mean()+(out['cos_loss'].mean() if 'cos_loss' in out else 0.)
@torch.no_grad()
def dit_eval(m,transport,x,text,seed):
    m.eval();total=0.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        for j in range(0,len(x),8):total+=float(losses(transport,m,x[j:j+8].cuda(),text[j:j+8]))*len(x[j:j+8])
    return total/len(x)
def train_dit(config,data,vae,run,fs):
    if not (run/'LATENTS.pt').exists():
        setseed(fs+100);zt=encode(vae,data['train']);zi=encode(vae,data['internal']);low=zt.amin((0,1),keepdim=True);high=zt.amax((0,1),keepdim=True)
        save(run/'LATENTS.pt',dict(train=zt,internal=zi,low=low,high=high))
    z=torch.load(run/'LATENTS.pt',map_location='cpu',weights_only=True);low,high=z['low'],z['high']
    xt=(2*(z['train']-low)/(high-low+1e-6)-1).permute(0,2,1).contiguous();xi=(2*(z['internal']-low)/(high-low+1e-6)-1).permute(0,2,1).contiguous()
    setseed(fs+200);m,changed=make_dit(config['seq_len']);ema=copy.deepcopy(m).eval()
    for p in ema.parameters():p.requires_grad_(False)
    write(run/'CLIP_RESTORATION.json',dict(overwritten_constructor_tensors=changed,restored_public_checkpoint=True,exact_restoration=True))
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4,betas=(.9,.95),weight_decay=0)
    transport=create_transport('Linear','velocity',use_cosine_loss=True,use_lognorm=True)
    best=float('inf');best_step=0;start=1;wall=time.monotonic()
    if (run/'dit_last.pt').exists():
        s=torch.load(run/'dit_last.pt',map_location='cpu',weights_only=False);m.load_state_dict(s['model']);ema.load_state_dict(s['ema']);opt.load_state_dict(s['optimizer']);restore(s['rng']);start=s['step']+1;best=s['best'];best_step=s['best_step']
    for step in range(start,100001):
        m.train();idx=torch.randint(0,len(xt),(32,));opt.zero_grad(set_to_none=True);summed=0.
        for j in range(0,32,8):
            ids=idx[j:j+8];loss=losses(transport,m,xt[ids].cuda(),[data['train_text'][i] for i in ids.tolist()])/4
            assert torch.isfinite(loss);loss.backward();summed+=float(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),m.parameters()):
                if b.requires_grad:a.mul_(.9999).add_(b,alpha=.0001)
        if step==1 or step%100==0:
            rec=dict(stage='DiT_training',step=step,max_steps=100000,loss=summed,best_step=best_step,best_internal_loss=best,seconds=time.monotonic()-wall,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
            write(run/'STATUS.json',rec)
            with (run/'DIT_CURVE.jsonl').open('a') as h:h.write(json.dumps(rec)+'\n')
            print('DIT_STEP',fs,step,summed,flush=True)
        if step%1000==0:
            val=dit_eval(ema,transport,xi,data['internal_text'],fs+901)
            if val<best:
                best=val;best_step=step;save(run/'dit_best.pt',dict(ema=ema.state_dict(),step=step,internal_loss=val))
            save(run/'dit_last.pt',dict(model=m.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),rng=rng(),step=step,best=best,best_step=best_step))
            with (run/'DIT_VALIDATION.jsonl').open('a') as h:h.write(json.dumps(dict(step=step,internal_loss=val))+'\n')
            if step>=10000 and step-best_step>=10000:break
    s=torch.load(run/'dit_best.pt',map_location='cpu',weights_only=False);ema.load_state_dict(s['ema']);ema.eval()
    write(run/'DIT_COMPLETE.json',dict(last_step=step,selected_step=s['step'],internal_loss=s['internal_loss'],test_reads=0))
    return ema,transport,low,high
@torch.no_grad()
def generate(model,transport,vae,parser,context,run,split,gs,low,high,L):
    from run_sparkov_argn_control import native_frame
    setseed(gs);parts=[];sampler=Sampler(transport).sample_ode(sampling_method='euler',num_steps=250)
    for j in range(0,len(context),8):
        rows=context.iloc[j:j+8];texts=['; '.join(f'{k}: {r[k]}' for k in rows.columns if k!='customer_id') for _,r in rows.iterrows()]
        sample=sampler(torch.randn(len(rows),32,L,device='cuda'),model,y=texts)[-1].permute(0,2,1).cpu()
        latent=(sample+1)/2*(high-low+1e-6)+low;decoded=vae.decoder(latent.cuda());eos=decoded['eos'].argmax(-1).cpu()
        tensor=torch.cat([decoded['bins'].gt(0).float(),torch.stack([q.argmax(-1) for q in decoded['cats']],-1).float(),decoded['nums']],-1).cpu()
        for k,(_,row) in enumerate(rows.iterrows()):
            hit=torch.nonzero(eos[k].eq(1)).flatten();length=int(hit[0])+1 if len(hit) else L
            back=parser.invert_fit(pd.DataFrame(tensor[k,:length].numpy(),columns=parser.column_name())).rename(columns={'fraud':'event_is_fraud','merchant':'receiver_or_mark','amount':'amount_or_numeric_value'})
            back['customer_id']=row.customer_id
            for name in rows.columns:
                if name!='customer_id':back[name]=row[name]
            parts.append(back)
        print('MELD_GENERATING',run.name,split,gs,min(j+8,len(context)),flush=True)
    raw=pd.concat(parts,ignore_index=True);d=native_frame(raw);target=run/f'generated_{split}_{gs}.parquet';d.to_parquet(target,index=False)
    write(run/f'GENERATION_{split}_{gs}.json',dict(rows=len(d),customers=d.entity_id.nunique(),sha256=digest(target),eos_native=True,max_cap=L,future_counts_supplied=False,test_reads=0))
def main():
    fs=int(sys.argv[1]);assert fs in [20261021,20261022];assert torch.cuda.is_available();torch.set_num_threads(2);tae.device='cuda';verify();setseed(fs)
    run=DATA/f'fit_{fs}';run.mkdir(exist_ok=True);assert not (run/'COMPLETE.json').exists()
    config=json.loads((DATA/'VAE_CONFIG.json').read_text());data=torch.load(DATA/'DATA.pt',map_location='cpu',weights_only=False)
    write(run/'START.json',dict(fit_seed=fs,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),config=config,test_reads=0))
    if (run/'VAE_COMPLETE.json').exists():
        if (run/'ADEQUACY_BLOCK.json').exists():return
        vae=tae.DeapStack(**config).cuda();vae.load_state_dict(torch.load(run/'vae_best.pt',map_location='cpu',weights_only=False)['model']);vae.eval()
    else:vae=train_vae(config,data,run,fs)
    if vae is None:return
    model,transport,low,high=train_dit(config,data,vae,run,fs)
    with (DATA/'parser.pkl').open('rb') as h:parser=pickle.load(h)
    internal=pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/workspace/OriginalData/ctx-data/part.000000-val.parquet')
    dev=pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/validation_context.parquet')
    generate(model,transport,vae,parser,internal,run,'internal',fs+902,low,high,config['seq_len'])
    for gs in [20261031,20261032]:generate(model,transport,vae,parser,dev,run,'development',gs,low,high,config['seq_len'])
    verify();write(run/'COMPLETE.json',dict(fit_seed=fs,vae=True,dit=True,development_generations=2,internal_generations=1,test_reads=0,adequacy_requires_independent_report=True))
if __name__=='__main__':main()
