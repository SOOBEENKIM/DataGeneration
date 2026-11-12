"""Bounded branch of frozen MeLD; original training source is never edited."""
import inspect, json, os, pickle, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/meld_500_pilot_v1'
sys.path.insert(0, str(ROOT / 'artifacts/comparison_completion_v1/local_source'))
import meld_train as base

def verify():
    base.verify()
    for p, h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():
        assert base.digest(p) == h, p

def bounded_source():
    s = inspect.getsource(base.train_dit)
    edits = [
        ('range(start,100001)', 'range(start,2001)'),
        ('max_steps=100000', 'max_steps=2000'),
        ('if step%1000==0:', 'if step%250==0:'),
        ('            if step>=10000 and step-best_step>=10000:break',
         '            if step in (500,1000): pilot_preview(step,m,ema,transport,vae,data,run,fs,low,high,config)'),
    ]
    for old, new in edits:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s

def evaluate_files(dst,split,files,fs,step,kind):
    # Separate interpreter prevents the vendor's `datasets` package from shadowing
    # Hugging Face datasets used by the frozen ARGN evaluator imports.
    dst.mkdir(parents=True,exist_ok=True)
    request=dst/'EVALUATION_REQUEST.json'
    base.write(request,dict(dst=str(dst),split=split,files=[(s,str(p)) for s,p in files],fs=fs,step=step,kind=kind))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=f'{ROOT}:{ROOT}/scripts:{OUT}/local_source')
    with (dst/'evaluation.log').open('a') as log:
        subprocess.run([sys.executable,str(OUT/'local_source/evaluate_pilot.py'),'--request',str(request)],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def pilot_preview(step, online, ema, transport, vae, data, run, fs, low, high, config):
    state = base.rng(); training = online.training
    try:
        context = base.pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/workspace/OriginalData/ctx-data/part.000000-val.parquet')
        with (base.DATA/'parser.pkl').open('rb') as f:
            parser = pickle.load(f)
        for kind, model in [('ema', ema)] + ([('online_diagnostic', online)] if step == 1000 else []):
            dst = run/'previews'/f'step_{step:06d}'/kind; dst.mkdir(parents=True, exist_ok=True)
            if (dst/'EVALUATED.json').exists():
                continue
            model.eval()
            gs = fs + 902
            base.write(dst/'PREVIEW.json', dict(step=step,kind=kind,fit_seed=fs,
                       selection_effect=False,preliminary=True,adequate_latest_baseline=False))
            base.generate(model,transport,vae,parser,context,dst,'internal',gs,low,high,config['seq_len'])
            evaluate_files(dst, 'internal', [(gs,dst/f'generated_internal_{gs}.parquet')], fs, step, kind)
    finally:
        base.restore(state); online.train(training)
    after = base.rng()
    assert base.torch.equal(state['cpu'], after['cpu'])
    assert all(base.torch.equal(a,b) for a,b in zip(state['cuda'],after['cuda']))
    assert state['python'] == after['python']
    assert state['numpy'][0] == after['numpy'][0] and base.np.array_equal(state['numpy'][1],after['numpy'][1])
    assert state['numpy'][2:] == after['numpy'][2:]
    base.write(run/f'PREVIEW_RNG_PARITY_{step}.json',dict(exact=True,step=step))

def main():
    verify()
    source = bounded_source()
    assert source == (OUT/'local_source/registered_train_dit.txt').read_text()
    if '--preflight' in sys.argv:
        print('PILOT_SOURCE_PREFLIGHT_OK'); return
    fs = int(sys.argv[1]); assert fs in [20261021,20261022]
    assert base.torch.cuda.is_available()
    base.torch.set_num_threads(2); base.tae.device='cuda'; base.setseed(fs)
    snapshot = OUT/'snapshots'/str(fs)
    capture = json.loads((snapshot/'SNAPSHOT.json').read_text())
    assert capture['epoch'] == 500
    for p,h in capture['hashes'].items(): assert base.digest(p)==h,p
    run = OUT/'runs'/str(fs); run.mkdir(parents=True,exist_ok=True)
    assert not (run/'COMPLETE.json').exists()
    config = json.loads((base.DATA/'VAE_CONFIG.json').read_text())
    data = base.torch.load(base.DATA/'DATA.pt',map_location='cpu',weights_only=False)
    selected = base.torch.load(snapshot/'vae_best.pt',map_location='cpu',weights_only=False)
    assert selected['epoch'] <= 500
    vae = base.tae.DeapStack(**config).cuda(); vae.load_state_dict(selected['model']); vae.eval()
    result = base.vae_eval(vae,data['internal'],fs+900)
    assert abs(result['reconstruction']-selected['internal']['reconstruction']) < 1e-4
    base.write(run/'VAE_SELECTION.json',dict(cap=500,selected_epoch=selected['epoch'],internal=result,
        source_hash=base.digest(snapshot/'vae_best.pt'),frozen=True))
    if not base.np.isfinite(result['reconstruction']) or result['confusion'][1][1]==0:
        base.write(run/'ADEQUACY_BLOCK.json',dict(stage='VAE',internal=result)); return
    base.write(run/'START.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],
               fit_seed=fs,vae_cap=500,dit_cap=2000,preliminary=True,test_reads=0))
    namespace = dict(base.__dict__); namespace['pilot_preview'] = pilot_preview
    exec(compile(source,str(OUT/'local_source/registered_train_dit.txt'),'exec'),namespace)
    model,transport,low,high = namespace['train_dit'](config,data,vae,run,fs)
    with (base.DATA/'parser.pkl').open('rb') as f: parser=pickle.load(f)
    contexts = {
      'internal': (base.pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/workspace/OriginalData/ctx-data/part.000000-val.parquet'),[fs+902]),
      'development': (base.pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/validation_context.parquet'),[20261031,20261032])}
    choice = json.loads((run/'DIT_COMPLETE.json').read_text())
    for split,(ctx,seeds) in contexts.items():
        files=[]
        for gs in seeds:
            flag=run/f'GENERATION_{split}_{gs}.json'; path=run/f'generated_{split}_{gs}.parquet'
            if flag.exists(): assert base.digest(path)==json.loads(flag.read_text())['sha256']
            else: base.generate(model,transport,vae,parser,ctx,run,split,gs,low,high,config['seq_len'])
            files.append((gs,path))
        evaluate_files(run/f'evaluation_{split}',split,files,fs,choice['selected_step'],'selected_ema')
    verify()
    for p,h in capture['hashes'].items(): assert base.digest(p)==h,p
    base.write(run/'COMPLETE.json',dict(fit_seed=fs,vae_cap=500,vae_selected=selected['epoch'],
       dit_cap=2000,dit_selected=choice['selected_step'],development_generations=2,previews=True,
       preliminary=True,final_comparison_complete=False,adequate_latest_baseline=False,test_reads=0))

if __name__=='__main__':
    main()
