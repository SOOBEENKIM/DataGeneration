"""Check exact-boundary capture, process targeting, unchanged computation, evaluator."""
import json,subprocess,sys,tempfile,time
from pathlib import Path
import torch
import numpy as np
from supervise import snapshot_files,identity,matches,stop,stopped,terminate,write
from evaluate_pilot import metrics
from pilot_train import ROOT,OUT,base,bounded_source

def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='meld500-preflight-') as tmp:
        src=Path(tmp)/'src';src.mkdir()
        torch.save(dict(epoch=500,best_epoch=495,best=.2,model={},optimizer={},rng={}),src/'vae_last.pt')
        torch.save(dict(epoch=495,internal=dict(reconstruction=.2),model={}),src/'vae_best.pt')
        (src/'VAE_CURVE.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,internal=dict(reconstruction=v))) for e,v in [(490,.3),(495,.2),(500,.21)])+'\n')
        write(src/'STATUS.json',dict(epoch=500));write(src/'START.json',dict(test=True))
        got=snapshot_files(src,Path(tmp)/'capture')
        assert got['epoch']==500 and got['best_epoch']==495
        checks['boundary_and_best_before_cap']=True
        saved=torch.load(src/'vae_last.pt',weights_only=False);saved['epoch']=501;torch.save(saved,src/'vae_last.pt')
        try:snapshot_files(src,Path(tmp)/'must_not_exist')
        except AssertionError:pass
        else:raise AssertionError('Overshot boundary not rejected')
        assert not (Path(tmp)/'must_not_exist').exists()
        checks['epoch501_not_relabelled']=True
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        try:
            time.sleep(.15);b=identity(p.pid);assert matches(b)
            forged=dict(b,start_ticks=str(int(b['start_ticks'])+1));assert not matches(forged)
            stop(b);assert stopped(b);terminate(b);p.wait(timeout=3);assert p.returncode==-15
        finally:
            if p.poll() is None:p.kill();p.wait()
        checks['pid_identity_suspend_terminate_owned_fixture']=True
    source=bounded_source();compile(source,'registered_train_dit','exec')
    assert 'range(start,2001)' in source and 'if step%250==0:' in source
    assert 'if step in (500,1000): pilot_preview' in source
    # All model/optimizer/EMA/sampling statements are inherited without rewriting.
    for marker in ["lr=2e-4,betas=(.9,.95)","a.mul_(.9999).add_(b,alpha=.0001)","setseed(fs+200)","torch.randint(0,len(xt),(32,))"]:
        assert marker in source
    checks['budget_only_training_source_changes']=True
    from run_sparkov_argn_control import native_frame
    internal=base.pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/workspace/OriginalData/tgt-data/part.000000-val.parquet')
    ctx=base.pd.read_parquet(ROOT/'artifacts/argn_state_first_v1/prepared/workspace/OriginalData/ctx-data/part.000000-val.parquet')
    assert set(internal.customer_id)==set(ctx.customer_id) and len(ctx)==69
    real=native_frame(internal);v,s=metrics(real,real)
    errors={k:x for k,x in v.items() if 'w1' in k}
    assert all(np.isfinite(x) and abs(x)<1e-14 for x in errors.values()),errors
    checks['internal69_mapping_and_identity_metrics']=errors
    from pilot_train import evaluate_files
    with tempfile.TemporaryDirectory(prefix='meld500-evaluator-') as tmp:
        path=Path(tmp)/'identity_fixture.parquet';real.to_parquet(path,index=False)
        dst=Path(tmp)/'evaluation'
        evaluate_files(dst,'internal',[(0,path)],0,0,'identity_fixture_not_generation')
        assert json.loads((dst/'EVALUATED.json').read_text())['complete']
    checks['separate_evaluator_interpreter']=True
    checks['no_model_fit_or_generation']=True
    write(OUT/'PREFLIGHT.json',dict(passed=True,checks=checks,test_reads=0))
    print('PREFLIGHT_OK',json.dumps(checks))

if __name__=='__main__':main()
