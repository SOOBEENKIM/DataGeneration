"""Isolate fine-gap information lost before neural training."""
import json
import shutil
from run_sparkov_argn_control import OUT,DOCS,LABEL,pd,np,seed,write,digest
from mostlyai.engine import analyze
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._encoding_types.tabular.numeric import encode_numeric,decode_numeric

base=OUT/'prepared';folder=OUT/'codec_digit_gap';folder.mkdir(parents=True,exist_ok=False)
for name in ['tgt-data','tgt-meta','ctx-data','ctx-meta']:
    shutil.copytree(base/'child/OriginalData'/name,folder/'OriginalData'/name)
meta=folder/'OriginalData/tgt-meta/encoding-types.json';types=json.loads(meta.read_text());types['gap']='TABULAR_NUMERIC_DIGIT';write(meta,types)
seed(20261200);analyze(workspace_dir=folder)
native=Workspace(base/'child').tgt_stats.read()['columns']['gap'];digit=Workspace(folder).tgt_stats.read()['columns']['gap']
assert digit['encoding_type']=='TABULAR_NUMERIC_DIGIT'
real=pd.read_parquet(base/'validation.parquet');x=real.gap.fillna(0);subsequent=real.event_index.gt(0).to_numpy();fraud=real[LABEL].eq('1').to_numpy();original=real.gap.to_numpy()
rows=[];errors=[]
for name,stats in [('native_binned',native),('digit',digit)]:
    encoded=encode_numeric(x,stats)
    for rs in range(20261200,20261210):
        seed(rs);decoded=decode_numeric(encoded,stats).to_numpy(dtype=float)
        error=np.abs(decoded[subsequent]-original[subsequent])
        errors.append(dict(codec=name,seed=rs,exact_fraction=float((error==0).mean()),median_absolute_seconds=float(np.median(error)),p99_absolute_seconds=float(np.quantile(error,.99)),max_absolute_seconds=float(error.max())))
        for threshold in [5,60,300,1800]:
            before=subsequent&(original>0)&(original<=threshold);after=subsequent&(decoded>0)&(decoded<=threshold)
            rows.append(dict(codec=name,seed=rs,threshold_seconds=threshold,original_events=int(before.sum()),decoded_events=int(after.sum()),intersection_events=int((before&after).sum()),membership_changed_events=int((before!=after).sum()),original_frauds=int((fraud&before).sum()),decoded_frauds=int((fraud&after).sum()),original_customers=real.loc[before,'entity_id'].nunique(),decoded_customers=real.loc[after,'entity_id'].nunique()))
pd.DataFrame(rows).to_csv(DOCS/'gap_codec_membership.csv',index=False);pd.DataFrame(errors).to_csv(DOCS/'gap_codec_errors.csv',index=False)
write(DOCS/'gap_codec_manifest.json',dict(native_stats=native,digit_stats=digit,training_partitions_unchanged=all(digest(p)==digest(folder/'OriginalData'/kind/p.name) for kind in ['tgt-data','ctx-data'] for p in (base/'child/OriginalData'/kind).glob('*.parquet')),outer_validation_used_to_fit_codec=False,neural_refit=False,script_sha256=digest(__file__),protocol_sha256=digest(DOCS/'CODEC_SENSITIVITY_PROTOCOL.md')))
print(pd.DataFrame(rows).groupby(['codec','threshold_seconds'])[['decoded_events','intersection_events','decoded_frauds']].agg(['min','mean','max']).to_string(),flush=True)
