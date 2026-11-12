"""Report the known clock regression control without changing the original study."""
from datetime import datetime,timezone
import json
from unittest.mock import patch
import pandas as pd
import report_argn_time_density as temporal
import diagnose_argn_time_density as diagnosis
from run_argn_clock_regression import ROOT,OUT,DOCS,CFG
from run_argn_clock_regression_amended import verify
from run_argn_time_density import digest,write

original_locate=temporal.locate


def locate(arm,fs,gs):
    if arm=='clock_joint':
        return OUT/f'runs/clock_joint_{fs}/generated_validation_{gs}.parquet',DOCS/f'evaluation/metrics_clock_joint_{fs}.csv'
    return original_locate(arm,fs,gs)


def figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#777777','#b34f62','#d29c2a','#458baa','#458266','#7750a6','#cf672c']
    labels=['Frozen candidate','Boundary head','Phase GMM','History density','History + clock','Clock-relative density','Conditional clock GMR']
    means=data.groupby(['arm','fit_seed']).mean(numeric_only=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm,label,color in zip(temporal.ARMS+['clock_joint'],labels,colors):
        x=means.loc[arm]
        for ax,field in zip(axes,['fraud_gap_seconds_log_w1','normal_stay_personal_gap_w1']):
            ax.scatter(x[field],x.onset_personal_gap_w1,color=color,s=32,alpha=.45)
            ax.scatter(x[field].mean(),x.onset_personal_gap_w1.mean(),color=color,marker='X',s=115,label=label)
    axes[0].set_xscale('log');axes[0].set_xlabel('All-fraud gap log-W1 (log axis)')
    axes[1].set_xlabel('Normal personal-gap W1');axes[1].legend(fontsize=8,loc='upper right')
    for ax in axes:ax.set_ylabel('Fraud-onset personal-gap W1');ax.grid(alpha=.2)
    fig.suptitle('Conditional time outputs: lower is better; dots = parent fits, X = mean')
    for ext in ['png','pdf','svg']:fig.savefig(DOCS/f'conditional_time_comparison.{ext}',dpi=180)
    plt.close(fig)
    x=pd.read_csv(temporal.DOCS/'clock_intervention/metrics.csv')
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for model,color,label in [('history_relative','#7750a6','Clock-relative density'),('clock_joint','#cf672c','Conditional clock GMR')]:
        for source,style in [('actual_clock','--'),('recursive_clock','-')]:
            g=x[x.model.eq(model)&x.clock_source.eq(source)].groupby('horizon').gap_w1.mean()
            ax.plot(g.index,g,style,marker='o',color=color,label=f'{label}: {source.replace("_"," ")}')
    ax.set(xlabel='Generated continuation length',ylabel='Gap log-W1',title='Same 105 customers, real prefix and oracle labels')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(DOCS/f'clock_feedback_intervention.{ext}',dpi=180)
    plt.close(fig)


def main():
    verify();assert (temporal.DOCS/'COMPLETE.json').exists()
    with patch.object(temporal,'locate',locate):result=temporal.collect(['clock_joint'])
    data,phases,personal,checks,sampling=result[:5]
    for name,frame in zip(['all_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit'],result[:5]):frame.to_csv(DOCS/f'{name}.csv',index=False)
    for row in checks.itertuples():
        p,_=locate(row.arm,row.fit_seed,row.generation_seed);g=json.loads((p.parent/f'GENERATION_{row.generation_seed}.json').read_text())
        assert digest(OUT/'time_head.pt')==g['head_sha256']
    combined=pd.concat([pd.read_csv(temporal.DOCS/'all_metrics.csv'),data],ignore_index=True)
    combined.to_csv(DOCS/'all_models.csv',index=False);combined.groupby(['fit_seed','arm']).mean(numeric_only=True).to_csv(DOCS/'fit_means.csv')
    with patch.object(temporal,'CFG',{**temporal.CFG,'arms':['clock_joint']}):gates=temporal.screen(combined)
    gates.to_csv(DOCS/'screen.csv',index=False)
    keys=['onset_personal_gap_w1','onset_gap_w1','fraud_gap_seconds_log_w1','normal_gap_seconds_log_w1','normal_stay_personal_gap_w1','fraud_ratio_log_w1','normal_amount_log_w1']
    lines=['# 개인 clock을 직접 조건으로 삼은 알려진 GMR 대조','',
        '같은2부모×4GPU draw 평균. 통계적 phase GMM4개를 한 번 적합하며 부모별 재학습이 아니다.',
        '조건부 Gaussian mixture 자체는 기존 방법이다. 최신 전체 생성기의 직접 재현과도 구분한다.','',
        '| 모델 | 개인 대비 시작 간격 | 시작 간격 | 전체 사기 간격 | 정상 간격 | 정상 개인 간격 | 개인 대비 사기 금액 | 정상 금액 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in temporal.ARMS+['clock_joint']:
        m=combined[combined.arm.eq(arm)].mean(numeric_only=True)
        lines.append('| '+temporal.DISPLAY.get(arm,'개인 속도 조건부 GMR')+' | '+' | '.join(f'{m[k]:.4f}' for k in keys)+' |')
    lines+=['','사전 비용 screen:',temporal.markdown_table(gates),'']
    (DOCS/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines))
    with patch.object(diagnosis,'ARMS',temporal.ARMS+['clock_joint']),patch.object(diagnosis,'locate',locate):diagnosis.main()
    figures(combined)
    write(DOCS/'COMPLETE.json',dict(statistical_phase_fits=4,new_neural_fits=0,new_full_generations=8,
        generated_rows=int(checks.rows.sum()),all_compared_generations=len(combined),test_events_read=False,
        source_sha256=digest(__file__),completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
