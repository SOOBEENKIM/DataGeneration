"""Final matched comparison, including unsuccessful progress-only ablations."""
from datetime import datetime,timezone
import json
from unittest.mock import patch
import pandas as pd
import report_argn_clock_evolution as first
import report_argn_time_density as prior
from run_argn_clock_location import ROOT,OUT,DOCS,CFG,verify,digest,write

original_locate=first.locate


def locate(arm,fs,gs):
    if arm in CFG['arms']:
        return OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet',DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    return original_locate(arm,fs,gs)


def figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=dict(frozen='Frozen ARGN extension',phase_mixture='Phase GMM',history_only='History density',clock_joint='Clock GMR',
        progress_joint='Progress GMR',progress_prefix='Progress + prefix fit',progress_rollout='Progress + rollout fit',
        clock_prefix='GMR + prefix location',clock_rollout='GMR + rollout location')
    colors=['#888888','#d29922','#287ca6','#c06522','#73509b','#699986','#bb8295','#214d37','#ad233a']
    means=data.groupby(['fit_seed','arm']).mean(numeric_only=True)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for (arm,label),color in zip(labels.items(),colors):
        d=means.xs(arm,level='arm')
        for ax,key,title in zip(axes,['onset_personal_gap_w1','fraud_gap_seconds_log_w1','normal_stay_personal_gap_w1'],
            ['Personal fraud-onset gap W1','All-fraud gap log-W1','Personal normal-gap W1']):
            ax.scatter(d.normal_gap_seconds_log_w1,d[key],s=23,color=color,alpha=.45)
            ax.scatter(d.normal_gap_seconds_log_w1.mean(),d[key].mean(),s=100,marker='X',color=color,label=label)
            ax.set(xlabel='All-normal gap log-W1',ylabel=title);ax.grid(alpha=.2)
    axes[2].legend(fontsize=7,loc='upper right');fig.suptitle('Single time output: normal timing and relationship costs (lower is better)')
    for ext in ['png','pdf','svg']:
        p=DOCS/f'clock_location_tradeoff.{ext}';fig.savefig(p,dpi=180)
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    verify();assert json.loads((DOCS/'DISPATCH_COMPLETE.json').read_text())['success']
    assert (first.DOCS/'COMPLETE.json').exists()
    with patch.object(first,'DOCS',DOCS),patch.object(first,'CFG',CFG),patch.object(first,'locate',locate):
        result=first.collect_cached(CFG['arms'])
    new,phases,personal,checks,sampling=result[:5]
    for row in checks.itertuples():
        p=locate(row.arm,row.fit_seed,row.generation_seed)[0];g=json.loads((p.parent/f'GENERATION_{row.generation_seed}.json').read_text())
        assert digest(OUT/row.arm/'time_head.pt')==g['head_sha256']
    for name,frame in zip(['new_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],result[:5]):frame.to_csv(DOCS/f'{name}.csv',index=False)
    previous=pd.read_csv(first.DOCS/'all_models.csv')
    for row in pd.read_csv(first.DOCS/'independent_checks.csv').itertuples():
        assert digest(locate(row.arm,row.fit_seed,row.generation_seed)[0])==row.output_sha256
    all_data=pd.concat([previous,new],ignore_index=True);all_data.to_csv(DOCS/'all_models.csv',index=False)
    arms=first.BASELINES+first.CFG['arms']+CFG['arms'];combined=all_data[all_data.arm.isin(arms)]
    combined.groupby(['fit_seed','arm'])[first.KEYS].mean().to_csv(DOCS/'fit_means.csv')
    combined.groupby('arm')[first.KEYS].agg(['mean','min','max']).to_csv(DOCS/'ranges.csv')
    with patch.object(first,'CFG',CFG):gates=first.screen(combined)
    means=combined.groupby(['fit_seed','arm']).mean(numeric_only=True)
    for i,row in gates.iterrows():
        gates.loc[i,'rollout_added_effect']=bool(row.arm=='clock_rollout' and row.passes_screen and
            means.loc[(row.fit_seed,row.arm)].normal_gap_seconds_log_w1<means.loc[(row.fit_seed,'clock_prefix')].normal_gap_seconds_log_w1)
    gates.to_csv(DOCS/'screen.csv',index=False)
    table=combined.groupby('arm')[first.KEYS[:7]].mean().reindex(arms).round(5).reset_index()
    lines=['# 기존 GMR 관계를 유지한 시간 위치 학습','',
        '같은2개 native 부모×4GPU draw. 본 대조는 두 위치 계수 적합2개이며 원래 GMR 가중치는 정확히 같다.',
        '앞선 진행 GMM3팔도 숨기지 않고 함께 비교한다. 모든 값은 development이며 최종 test는 미사용이다.','',
        prior.markdown_table(table),'','부모별 사전 비용 판정:','',prior.markdown_table(gates),'']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
    with patch.object(first,'DOCS',DOCS),patch.object(first,'CFG',CFG),patch.object(first,'locate',locate):first.trajectory(combined)
    figure(combined)
    write(DOCS/'COMPLETE.json',dict(new_full_generations=len(new),generated_rows=int(checks.rows.sum()),
        current_cycle_new_full_generations=len(new)+24,current_cycle_rows=int(checks.rows.sum())+int(json.loads((first.DOCS/'COMPLETE.json').read_text())['generated_rows']),
        new_gmm_fits=0,new_calibration_fits=2,new_native_argn_fits=0,focused_matched_generations=len(combined),
        all_preserved_generations=len(all_data),test_events_read=False,source_sha256=digest(__file__),completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
