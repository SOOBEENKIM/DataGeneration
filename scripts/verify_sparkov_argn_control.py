"""Independent artifact checks; model errors are measured, never filtered away."""
import argparse
import inspect
import json
import re
import shutil
from pathlib import Path
from run_sparkov_argn_control import OUT,DOCS,CFG,CONFIG,CORE,STATIC,LABEL,MERCHANT,pd,np,write,digest

def verify(argn_only=False):
    from mostlyai.engine._workspace import Workspace
    import mostlyai.engine._tabular.training as installed_training
    base=OUT/'prepared';manifest=json.loads((DOCS/'preparation.json').read_text())
    assert manifest['config_sha256']==digest(CONFIG) and not manifest['test_events_loaded']
    train=pd.read_parquet(base/'train.parquet');val=pd.read_parquet(base/'validation.parquet');rt=pd.read_parquet(base/'validation_roundtrip.parquet')
    assert (len(train),train.entity_id.nunique(),len(val),val.entity_id.nunique())==(916567,688,177997,147)
    assert set(train.entity_id).isdisjoint(val.entity_id)
    pd.testing.assert_frame_equal(val[[MERCHANT,'category',LABEL]],rt[[MERCHANT,'category',LABEL]],check_dtype=False)
    ws=Workspace(base/'child');stats=ws.tgt_stats.read();ctxstats=ws.ctx_stats.read()
    assert set(stats['columns'])==set(CORE) and set(ctxstats['columns'])==set(STATIC)
    assert (stats['no_of_training_records'],stats['no_of_validation_records'])==(619,69)
    trn=pd.read_parquet(base/'child/OriginalData/tgt-data/part.000000-trn.parquet')
    check=pd.read_parquet(base/'child/OriginalData/tgt-data/part.000000-val.parquet')
    assert set(trn.customer_id).isdisjoint(check.customer_id)
    assert set(trn.customer_id)|set(check.customer_id)==set(train.entity_id)
    reference=DOCS/'provenance';reference.mkdir(exist_ok=True)
    fit_rows=[]
    for s in CFG['fit_seeds']:
        native=None
        for arm in ['native','relaxed']:
            folder=OUT/'runs'/f'{arm}_{s}';current=Workspace(folder/'workspace');fit=json.loads((folder/'FIT.json').read_text())
            assert digest(current.model_tabular_weights_path)==fit['weights_sha256']
            assert current.tgt_stats.read()==stats and current.ctx_stats.read()==ctxstats
            p=folder/'workspace/ModelStore/model-data/progress-messages.csv';progress=pd.read_csv(p)
            comparable=progress.drop(columns='total_time')
            if arm=='native':
                assert progress.epoch.max()==14;native=comparable
            else:
                pd.testing.assert_frame_equal(native.reset_index(drop=True),comparable[comparable.epoch.le(14)].reset_index(drop=True))
                source=json.loads((folder/'source_control.json').read_text())
                assert source['official_sha256']==digest(inspect.getfile(installed_training))
                shutil.copy2(folder/'training_cap_control.diff',reference/f'{arm}_{s}_source.diff')
                shutil.copy2(folder/'source_control.json',reference/f'{arm}_{s}_source.json')
            last=progress.iloc[-1];best=progress[progress.is_checkpoint.eq(1)].iloc[-1]
            fit_rows.append(dict(run=f'{arm}_{s}',last_epoch=float(last.epoch),selected_epoch=float(best.epoch),selected_validation_loss=float(best.val_loss),updates=int(last.steps),fit_seconds=fit['seconds'],weights_sha256=fit['weights_sha256'],stop_reason='native_customer_count_cap' if arm=='native' else 'native_validation_early_stopping'))
            shutil.copy2(p,reference/f'{arm}_{s}_progress.csv')
        folder=OUT/'runs'/f'digit_relaxed_{s}';current=Workspace(folder/'workspace');fit=json.loads((folder/'FIT.json').read_text())
        assert digest(current.model_tabular_weights_path)==fit['weights_sha256']
        digit_stats=current.tgt_stats.read()
        assert digit_stats['columns']['gap']['encoding_type']=='TABULAR_NUMERIC_DIGIT'
        assert all(stats['columns'][c]==digit_stats['columns'][c] for c in CORE if c!='gap')
        assert current.ctx_stats.read()==ctxstats
        p=folder/'workspace/ModelStore/model-data/progress-messages.csv';progress=pd.read_csv(p);last=progress.iloc[-1];best=progress[progress.is_checkpoint.eq(1)].iloc[-1]
        fit_rows.append(dict(run=f'digit_relaxed_{s}',last_epoch=float(last.epoch),selected_epoch=float(best.epoch),selected_validation_loss=float(best.val_loss),updates=int(last.steps),fit_seconds=fit['seconds'],weights_sha256=fit['weights_sha256'],stop_reason='native_validation_early_stopping' if last.epoch<100 else 'registered_100_epoch_cap'))
        shutil.copy2(p,reference/f'digit_relaxed_{s}_progress.csv')
        source=json.loads((folder/'source_control.json').read_text());assert source['official_sha256']==digest(inspect.getfile(installed_training))
        shutil.copy2(folder/'source_control.json',reference/f'digit_relaxed_{s}_source.json')
        folder=OUT/'runs'/f'numeric_digit_relaxed_{s}';current=Workspace(folder/'workspace');fit=json.loads((folder/'FIT.json').read_text())
        assert digest(current.model_tabular_weights_path)==fit['weights_sha256']
        both_stats=current.tgt_stats.read()
        assert both_stats['columns']['gap']==digit_stats['columns']['gap']
        assert both_stats['columns']['amount_or_numeric_value']['encoding_type']=='TABULAR_NUMERIC_DIGIT'
        assert all(stats['columns'][c]==both_stats['columns'][c] for c in CORE if c not in ['gap','amount_or_numeric_value'])
        assert current.ctx_stats.read()==ctxstats
        p=folder/'workspace/ModelStore/model-data/progress-messages.csv';progress=pd.read_csv(p);last=progress.iloc[-1];best=progress[progress.is_checkpoint.eq(1)].iloc[-1]
        fit_rows.append(dict(run=f'numeric_digit_relaxed_{s}',last_epoch=float(last.epoch),selected_epoch=float(best.epoch),selected_validation_loss=float(best.val_loss),updates=int(last.steps),fit_seconds=fit['seconds'],weights_sha256=fit['weights_sha256'],stop_reason='native_validation_early_stopping' if last.epoch<100 else 'registered_100_epoch_cap'))
        shutil.copy2(p,reference/f'numeric_digit_relaxed_{s}_progress.csv')
        source=json.loads((folder/'source_control.json').read_text());assert source['official_sha256']==digest(inspect.getfile(installed_training))
        shutil.copy2(folder/'source_control.json',reference/f'numeric_digit_relaxed_{s}_source.json')
    for row in fit_rows:
        log=(OUT/'logs'/f"{row['run']}.log").read_text()
        parameters=set(re.findall(r'no_of_model_params=(\d+)',log))
        assert len(parameters)==1
        row['model_parameters']=int(next(iter(parameters)))
    expected=[]
    for s in CFG['fit_seeds']:
        for arm in ['native','relaxed','digit_relaxed','numeric_digit_relaxed']:
            for pop in ['validation','synthetic']:
                for gs in CFG['generation_seeds']:expected.append(OUT/'runs'/f'{arm}_{s}'/f'generated_{pop}_{gs}.parquet')
        for priority in ['category',LABEL]:
            folder=OUT/'runs'/f'order_{priority}_relaxed_{s}'
            original=json.loads((OUT/'runs'/f'relaxed_{s}/FIT.json').read_text())['weights_sha256']
            assert json.loads((folder/'FIT.json').read_text())['weights_sha256']==original
            for gs in CFG['generation_seeds']:
                path=folder/f'generated_validation_{gs}.parquet'
                reference_data=pd.read_parquet(OUT/'runs'/f'relaxed_{s}'/path.name)
                alternative=pd.read_parquet(path)
                pd.testing.assert_series_equal(reference_data.groupby('entity_id').size(),alternative.groupby('entity_id').size())
                expected.append(path)
        for arm in ['digit_relaxed','numeric_digit_relaxed']:
            folder=OUT/'runs'/f'order_category_{arm}_{s}'
            original=json.loads((OUT/'runs'/f'{arm}_{s}/FIT.json').read_text())['weights_sha256']
            assert json.loads((folder/'FIT.json').read_text())['weights_sha256']==original
            for gs in CFG['generation_seeds']:
                path=folder/f'generated_validation_{gs}.parquet'
                reference_data=pd.read_parquet(OUT/'runs'/f'{arm}_{s}'/path.name);alternative=pd.read_parquet(path)
                pd.testing.assert_series_equal(reference_data.groupby('entity_id').size(),alternative.groupby('entity_id').size())
                expected.append(path)
    for arm in ['row_resampling','transition_resampling']:
        for gs in CFG['generation_seeds']:expected.append(OUT/'runs'/arm/f'generated_validation_{gs}.parquet')
    if not argn_only:
        for gs in CFG['generation_seeds']:expected.append(OUT/'runs/cpar_20260928'/f'generated_supported_validation_{gs}.parquet')
    outputs=[]
    for p in expected:
        d=pd.read_parquet(p)
        assert set(d.columns)=={'entity_id','event_index',*CORE}
        assert not d.duplicated(['entity_id','event_index']).any()
        np.testing.assert_array_equal(d.event_index,d.groupby('entity_id',sort=False).cumcount())
        is_supported='generated_supported_validation_' in p.name
        assert d.entity_id.nunique()==(143 if is_supported else 147)
        context=base/('supported_validation_context.parquet' if is_supported else 'synthetic_context.parquet' if 'generated_synthetic_' in p.name else 'validation_context.parquet')
        assert set(d.entity_id)==set(pd.read_parquet(context).customer_id)
        assert d.loc[d.event_index.eq(0),'gap'].isna().all()
        outputs.append(dict(path=str(p.relative_to(OUT)),sha256=digest(p),events=len(d),customers=d.entity_id.nunique()))
    if not argn_only:
        folder=OUT/'runs/cpar_20260928';fit=json.loads((folder/'FIT.json').read_text());history=pd.read_csv(folder/'history.csv')
        assert len(history)==128 and np.isfinite(history.Loss).all()
        assert digest(folder/'model.pkl')==fit['model_sha256']
        coverage=json.loads((DOCS/'cpar_context_encoder_check.json').read_text())
        assert coverage['requested_customers']==147 and coverage['supported_customers']==143
        assert coverage['model_sha256']==fit['model_sha256']
        assert coverage['observed_support_matches_registered_cohort']
        shutil.copy2(folder/'START.json',reference/'cpar_start.json');shutil.copy2(folder/'FIT.json',reference/'cpar_fit.json');shutil.copy2(folder/'history.csv',reference/'cpar_history.csv')
    pd.DataFrame(fit_rows).to_csv(DOCS/'fit_summary.csv',index=False)
    write(DOCS/('verification_argn.json' if argn_only else 'verification.json'),dict(status='PASS',scope='ARGN and descriptive controls only; CPAR pending' if argn_only else 'all registered primary and diagnostic outputs',outputs=outputs,paired_first14_progress_equal=True,generation_order_planned_lengths_equal=True,digit_control_other_column_statistics_equal=True,installed_argn_training_unchanged=True,outer_validation_excluded_from_fit=True,test_events_loaded=False,verification_script_sha256=digest(__file__)))
    print(f'PASS: {len(expected)} output datasets; paired fit prefixes match exactly.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--argn-only',action='store_true');verify(p.parse_args().argn_only)
