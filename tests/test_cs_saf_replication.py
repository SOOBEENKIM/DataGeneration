import math
from pathlib import Path
import numpy as np
import pytest
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cs_saf_replication import (
    ROOT, CANDIDATES, BANK_PARAMETERS, ReplicationU, initialize_banks,
    load_contract, trial_config, load_cache, parse_cell, make_model,
    verify_initialization, tensor_digest, paired_contrasts, seed_statistics, expansion_gate,
)
from experiments.cs_saf_pilot import batch, subset, state_digest
from models.cs_saf_v2 import CSSAFv2
from scripts.run_cs_saf_replication import available_gpus, authorize_cell


@pytest.fixture(scope='module')
def payload():
    torch.set_num_threads(1)
    p=load_cache(ROOT/'artifacts/cs_saf/prepared_v1/pi_0.05_kappa_1.pt')
    return dict(p,train=subset(p['train'],4),validation=subset(p['validation'],2))


def test_every_trial_varies_all_random_initialization_with_paired_common_tensors(payload):
    common=[];banks=[]
    for trial_index in range(5):
        trial,_=trial_config(trial_index);states=[];rng=[];checks=[]
        for c in CANDIDATES:
            torch.manual_seed(trial['model_seed'])
            m=make_model(payload,c,torch.device('cpu'),trial_index)
            states.append(m.state_dict());rng.append(torch.get_rng_state())
            checks.append(verify_initialization(m,payload,trial_index))
        assert len({x['common_state_sha256'] for x in checks})==1
        assert all(torch.equal(states[0][k],s[k]) for s in states[1:] for k in states[0])
        assert all(torch.equal(states[1][k],states[2][k]) for k in states[1])
        assert all(torch.equal(rng[0],x) for x in rng[1:])
        common.append(checks[0]['common_state_sha256']);banks.append(checks[0]['bank_state_sha256'])
    assert len(set(common))==len(set(banks))==5


def test_bank_initialization_matches_original_distribution_and_does_not_consume_global_rng(payload):
    s=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    m=CSSAFv2('CS2-U1',s.gap_support,**s.model_config_kwargs())
    expected={k:v.clone() for k,v in m.state_dict().items()}
    rng=torch.get_rng_state();initialize_banks(m,20261010)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,expected[k]) for k,v in m.state_dict().items())
    initialize_banks(m,20261101)
    assert any(not torch.equal(m.state_dict()[k],expected[k]) for k in BANK_PARAMETERS)
    for k,fanin in zip(BANK_PARAMETERS,(136,136,32)):
        assert getattr(m,k).abs().max()<=1/math.sqrt(fanin)


def test_U_forward_training_objective_gradients_and_ordinary_audit_unchanged(payload):
    s=SAFTensorizerState.from_dict(payload['tensorizer_state'])
    old=CSSAFv2('CS2-U1',s.gap_support,**s.model_config_kwargs())
    new=ReplicationU('CS2-U1',s.gap_support,**s.model_config_kwargs())
    with torch.no_grad():old.route_interaction_weight.uniform_(-.5,.5)
    new.load_state_dict(old.state_dict())
    x=batch(payload['train'],torch.arange(4),torch.device('cpu'))
    a=old.loss_terms(**x);b=new.loss_terms(**x)
    assert all(torch.equal(a[k],b[k]) for k in a)
    args=dict(train_entities=4,transition_counts_by_code={3:20,4:30})
    la=old.objective(a,**args);lb=new.objective(b,**args)
    assert all(torch.equal(x,y) for x,y in zip(la,lb))
    la[0].backward();lb[0].backward()
    assert all((x.grad is None and y.grad is None) or torch.equal(x.grad,y.grad)
               for x,y in zip(old.parameters(),new.parameters()))
    ctx=torch.randn(137,136);codes=torch.tensor([3,4]*68+[3]);prev=torch.full((137,),3)
    assert all(torch.equal(x,y) for x,y in zip(old.response_curves(ctx,prev,static_codes=codes),new.response_curves(ctx,prev,static_codes=codes)))
    q,repeat=new.response_curves(ctx,prev,zero_gap=True,static_codes=codes)
    assert torch.count_nonzero(q-q[:,:1])==torch.count_nonzero(repeat-repeat[:,:1])==0
    torch.testing.assert_close(q[:,0],old.copy_logits(ctx,torch.ones(137),zero_gap=True,static_codes=codes).sigmoid())


def test_seed_intervals_use_training_trials_not_entity_sample_size():
    s=seed_statistics([-2.,-1.,0.,1.,2.])
    assert s['n_trials']==5 and s['mean']==0 and s['sample_SD']==pytest.approx(math.sqrt(2.5))
    assert s['standard_error']==pytest.approx(math.sqrt(.5))
    assert s['descriptive_95_percent_t_interval']==pytest.approx([-2.7764451051977987/math.sqrt(2),2.7764451051977987/math.sqrt(2)])
    assert s['negative_count']==2 and s['nonpositive_count']==3
    assert seed_statistics([1.,2.,3.])['descriptive_95_percent_t_interval'] is None
    with pytest.raises(ValueError):seed_statistics([1.]*6)
    with pytest.raises(ValueError):seed_statistics([float('nan')])


@pytest.mark.parametrize('response_count,null,active,expected',[
    (5,-.001,0.,'PASS'),(4,-.001,-.001,'FAIL'),(5,0.,-.001,'FAIL'),(5,-.001,.00001,'FAIL')])
def test_gate_requires_all_response_seeds_and_both_accuracy_directions(response_count,null,active,expected):
    gates=[{'decision':'PASS' if i<response_count else 'FAIL'} for i in range(5)]
    effects={c:{'null':[null]*5,'active':[active]*5} for c in ('CS3-E1','CS2-U1')}
    assert expansion_gate(gates,effects)['decision']==expected
    if expected=='PASS':
        effects['CS2-U1']['active']=[.01]*5
        assert expansion_gate(gates,effects)['decision']=='FAIL'
    with pytest.raises(ValueError):expansion_gate(gates[:4],effects)


def test_paired_contrasts_preserve_missing_snapshot_and_reject_misalignment(tmp_path):
    folders={};ids='best_validation_label_1_entity_ids';key=ids.replace('_entity_ids','_metrics')
    for i,c in enumerate(CANDIDATES):
        folder=tmp_path/c;folder.mkdir();folders[c]=folder
        fields={ids:['a','b'],key:np.full((2,7),float(i))}
        if c!='CS3-E1':fields.update({ids.replace('best','epoch_9'):['a','b'],key.replace('best','epoch_9'):np.full((2,7),float(i))})
        np.savez(folder/'accuracy_arrays.npz',**fields)
    result=paired_contrasts(folders)
    assert result['CS4-ER1_minus_CS3-E1'][key]['grid_mark_TV']['mean']==1
    assert key.replace('best','epoch_9') not in result['CS4-ER1_minus_CS3-E1']
    assert key.replace('best','epoch_9') in result['CS4-ER1_minus_CS2-U1']
    np.savez(folders['CS2-U1']/'accuracy_arrays.npz',**{ids:['b','a'],key:np.zeros((2,7))})
    with pytest.raises(ValueError,match='alignment'):paired_contrasts(folders)


def test_busy_GPU_excluded_even_when_utilization_not_observed(monkeypatch):
    def output(command,**kwargs):
        if '--query-gpu=index,uuid,memory.free' in command:return '0, gpu-a, 24000\n1, gpu-b, 24000\n2, gpu-c, 1000\n'
        return 'gpu-b, 12345\n'
    monkeypatch.setattr('scripts.run_cs_saf_replication.subprocess.check_output',output)
    assert list(available_gpus([0,1,2]))==[0]


def test_registered_scope_and_expansion_require_positive_stage1_evidence(tmp_path):
    c,cfg=load_contract()
    assert c['stage1']['scientific_fits']==30 and c['stage2']['additional_scientific_fits']==90
    assert c['excluded_discovery_model_seed'] not in [t['model_seed'] for t in c['trials']]
    assert cfg['training']['epochs']==50 and not c['data']['test_access']
    for pi in (.05,.10,.25,.50):assert parse_cell(Path(f'pi_{pi:.2f}_kappa_1.pt'))==(pi,1)
    with pytest.raises(ValueError):parse_cell(Path('pi_0.15_kappa_1.pt'))
    with pytest.raises(ValueError):trial_config(5)
    with pytest.raises(FileNotFoundError):authorize_cell(Path('pi_0.10_kappa_1.pt'),tmp_path)


def test_technical_failure_stops_new_dispatch_and_waits_for_inflight(tmp_path,monkeypatch):
    import scripts.run_cs_saf_replication as runner
    launched=[];polls=[]
    class Process:
        def __init__(self,number):self.pid=100+number;self.number=number;self.calls=0
        def poll(self):
            self.calls+=1;polls.append(self.number)
            return 1 if self.number==0 else (0 if self.calls>=2 else None)
    def launch(command,**kwargs):
        process=Process(len(launched));launched.append(process);return process
    monkeypatch.setattr(runner,'frozen_source',lambda:'frozen')
    monkeypatch.setattr(runner,'available_gpus',lambda ids:{g:{'uuid':f'gpu-{g}'} for g in ids})
    monkeypatch.setattr(runner.subprocess,'Popen',launch)
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(runner,'collect_job',lambda folder,source:{'status':'COMPLETE'})
    with pytest.raises(RuntimeError,match='technical failure'):
        runner.execute_stage(.05,tmp_path/'cache',tmp_path/'cpu.json',tmp_path,[0,1])
    assert len(launched)==2 and launched[1].calls>=2
    assert (tmp_path/'pi_0.05/FAILED.json').exists()
