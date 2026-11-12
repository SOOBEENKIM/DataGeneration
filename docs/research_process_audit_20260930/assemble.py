"""Summarize already verified rows; never reopen training or test events."""
from pathlib import Path
import json
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
raw=pd.read_csv(OUT/'RAW_RECOMPUTATION.csv')
metric=[c for c in raw if c not in ['arm','fit_seed','generation_seed']]
for frame,name in [
    (raw[raw.arm.isin(['clock_prefix','current','init_only','init_handoff'])],
     'PRESERVED_2PARENTS_4DRAWS.csv'),
    (raw[(raw.fit_seed==20260930)&raw.generation_seed.isin([20261011,20261012])],
     'LATEST_MATCHED_1PARENT_2DRAWS.csv')]:
    table=frame.groupby('arm')[metric].mean()
    table['parent_count']=frame.groupby('arm').fit_seed.nunique()
    table['draw_count']=frame.groupby('arm').size()
    table.to_csv(OUT/name)

old=pd.read_csv(ROOT/'artifacts/argn_clock_feedback_gain_v1/report/all_models.csv')
cols=['normal_gap_seconds_log_w1','onset_personal_gap_w1',
      'normal_stay_personal_gap_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']
arms=['history_only','clock_joint','clock_prefix','current','init_only','init_handoff']
old[old.arm.isin(arms)].groupby(['arm','fit_seed'])[cols].mean().to_csv(OUT/'HISTORICAL_PER_PARENT.csv')
effects=[]
base=old[old.arm=='init_only'].set_index(['fit_seed','generation_seed'])
for arm in arms:
    if arm=='init_only':continue
    other=old[old.arm==arm].set_index(['fit_seed','generation_seed'])
    for metric in cols:
        delta=base[metric]-other[metric]
        effects.append(dict(comparator=arm,metric=metric,delta_init_minus_comparator=delta.mean(),
          lower_draws=int((delta<0).sum()),total_draws=len(delta),
          lower_parent_means=int((delta.groupby(level=0).mean()<0).sum()),
          parent_count=delta.index.get_level_values(0).nunique(),
          source='previous_saved_metrics; not additional independent head fits'))
pd.DataFrame(effects).to_csv(OUT/'HISTORICAL_PAIRED_EFFECTS.csv',index=False)
metadata=dict(
    summary_only_amendment='Removed averaged seed-ID columns; added parent/draw counts. No metric changed.',
    raw_recomputation= '40 saved outputs, 440 checks; see AUDIT_COMPLETE.json',
    prior_summary_source='argn_clock_feedback_gain_v1/report/all_models.csv',
    caution='History/GMR comparative rows use prior audited metric tables, not extra raw recomputation this turn.',
    independent_new_head_refits=0,new_fit=0,new_generation=0,test_events_read=False)
(OUT/'ASSEMBLY_NOTE.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
