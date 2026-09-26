"""Replace only the remaining amount codec using official native analysis."""
import json
import shutil
from run_sparkov_argn_control import OUT,DOCS,CORE,AMOUNT,LABEL,pd,np,seed,write,digest
from mostlyai.engine import analyze,encode
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._encoding_types.tabular.numeric import encode_numeric,decode_numeric

base=OUT/'codec_digit_gap';folder=OUT/'codec_digit_both';folder.mkdir(parents=True,exist_ok=False)
for name in ['tgt-data','tgt-meta','ctx-data','ctx-meta']:shutil.copytree(base/'OriginalData'/name,folder/'OriginalData'/name)
meta=folder/'OriginalData/tgt-meta/encoding-types.json';types=json.loads(meta.read_text());types[AMOUNT]='TABULAR_NUMERIC_DIGIT';write(meta,types)
seed(20261300);analyze(workspace_dir=folder)
ws=Workspace(folder);original=Workspace(base).tgt_stats.read();analysis=ws.tgt_stats.read()
write(folder/'independent_analysis_not_used_in_full.json',analysis)
frozen=json.loads(json.dumps(original));frozen['columns'][AMOUNT]=analysis['columns'][AMOUNT];ws.tgt_stats.write(frozen)
shutil.copy2(base/'ModelStore/ctx-stats/stats.json',folder/'ModelStore/ctx-stats/stats.json')
assert frozen['columns'][AMOUNT]['encoding_type']=='TABULAR_NUMERIC_DIGIT'
encode(workspace_dir=folder)
real=pd.read_parquet(OUT/'prepared/validation.parquet');rows=[]
for c in ['gap',AMOUNT]:
    stats=frozen['columns'][c];x=real[c].fillna(0) if c=='gap' else real[c]
    encoded=encode_numeric(x,stats)
    for rs in range(20261300,20261310):
        seed(rs);decoded=decode_numeric(encoded,stats).to_numpy(dtype=float)
        for label in ['all','0','1']:
            use=np.ones(len(real),bool) if label=='all' else real[LABEL].eq(label).to_numpy()
            if c=='gap':use &=real.event_index.gt(0).to_numpy()
            before=real[c].to_numpy()[use];after=decoded[use];error=np.abs(after-before)
            rows.append(dict(field=c,label=label,seed=rs,events=int(use.sum()),original_mean=float(before.mean()),decoded_mean=float(after.mean()),original_p99=float(np.quantile(before,.99)),decoded_p99=float(np.quantile(after,.99)),exact_fraction=float((error==0).mean()),max_absolute_error=float(error.max())))
pd.DataFrame(rows).to_csv(DOCS/'numeric_digit_roundtrip.csv',index=False)
write(DOCS/'numeric_digit_preparation.json',dict(protocol_sha256=digest(DOCS/'AMOUNT_SCALE_PROTOCOL.md'),changed_column_from_gap_digit=AMOUNT,other_columns_equal=all(original['columns'][c]==frozen['columns'][c] for c in CORE if c!=AMOUNT),context_equal=Workspace(base).ctx_stats.read()==ws.ctx_stats.read(),outer_validation_used_to_fit=False,stats=frozen,script_sha256=digest(__file__)))
print(pd.DataFrame(rows).groupby(['field','label'])[['original_mean','decoded_mean','exact_fraction']].mean().to_string(),flush=True)
