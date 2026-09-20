"""Compact tables and scientific figures from completed, verified results."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cs_saf_rollout_calibration_v1 import DOC


def main():
    assert json.loads((DOC/'verification.json').read_text())['status']=='PASS'
    co=pd.read_csv(DOC/'conditional_metrics.csv',dtype={'group':str})
    ge=pd.read_csv(DOC/'generation_by_parent.csv',dtype={'group':str})
    costs=pd.read_csv(DOC/'costs_by_parent.csv',dtype={'group':str})
    oracle=pd.read_csv(DOC/'oracle_repeat_summary.csv',dtype={'group':str})
    decisions=json.loads((DOC/'decision.json').read_text())
    active=ge[(ge.kappa==1)&(ge.group=='1')].merge(co[(co.kappa==1)&(co.group=='1')],
        on=['kappa','trial','name','variant','group'],suffixes=('','_conditional'))
    result=active.groupby(['name','variant'])[['short_gap_repeat_curve_l1','repeat_brier','mark_nll']].mean()
    lines=['# 이번 보정 비교의 전체 결과 표','',
        '낮을수록 좋다. 생성 점수는 부모별 세 생성 난수를 먼저 평균한 뒤 세 학습 시드를 평균했다. 같은 인공 데이터 seed42·집단 비율 10%의 탐색 결과다.','',
        '## 관계가 있는 집단의 평균','',
        '| 모델 | 보정 | 생성 관계 L1 | 반복 Brier | 행동 NLL |','|---|---|---:|---:|---:|']
    for (name,method),r in result.iterrows():
        lines.append(f'| {name} | {method} | {r.short_gap_repeat_curve_l1:.6f} | {r.repeat_brier:.6f} | {r.mark_nll:.6f} |')
    lines.extend(['','## 사전 판정','',
        '| 비교 | 생성 개선 | 예측 비용 | 기본 예측 | 다른 분포 | 비활성 생성 | 비활성 반응 | 동시 통과 |',
        '|---|---|---|---|---|---|---|---|'])
    for d in decisions['comparisons']:
        keys=['generation_improvement_pass','prediction_cost_pass','basic_prediction_pass','distribution_pass','null_generation_pass','null_response_pass','joint_pass']
        lines.append(f'| {d["name"]}: {d["variant"]}−A | '+' | '.join('통과' if d[k] else '**실패**' for k in keys)+' |')
    lines.extend(['','## 학습 시드별 활성 생성 결과','',
        '| 모델 | 보정 | 시드 0 | 시드 1 | 시드 2 | A 대비 평균 감소 | 개선 시드 |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for d in decisions['comparisons']:
        lines.append(f'| {d["name"]} | {d["variant"]} | '+' | '.join(f'{x:.6f}' for x in d['candidate_values'])+
            f' | {d["relative_reduction"]*100:.2f}% | {d["improving_seeds"]}/3 |')
    lines.extend(['','## 비용의 최댓값','',
        '예측·분포 비용은 각 조건/집단/학습 시드에서 A 대비 증가한 값이다. 반응 비용은 비활성 조건만 포함한다. 서로 다른 열의 최댓값은 다른 조건에서 발생할 수 있다.','',
        '| 모델 | 보정 | Brier 증가 | NLL 증가 | gap KS 증가 | 행동 TV 증가 | 수치값 KS 증가 | 비활성 반응 증가 |',
        '|---|---|---:|---:|---:|---:|---:|---:|'])
    for (name,m),f in costs.groupby(['name','variant']):
        null=f[(f.kappa==0)|(f.group=='0')]
        values=[f.delta_repeat_brier.max(),f.delta_mark_nll.max(),f.delta_gap_ks.max(),f.delta_mark_sparse_tv.max(),f.delta_amount_ks.max(),null.delta_fixed_history_range.max()]
        lines.append(f'| {name} | {m} | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    lines.extend(['','## 같은 표본 수의 정답 생성기 참고값','',
        '각 행은 생성 난수 30개의 분포다. 5–95% 구간은 이 고정 reference에서 생성 표본에 따른 변동이며 신뢰구간·이론적 하한·새 데이터에서의 독립 재현을 뜻하지 않는다.','',
        '| 조건 | 집단 | 생성기 | reference | L1 평균 | 5% | 중앙값 | 95% |',
        '|---|---|---|---|---:|---:|---:|---:|'])
    for _,r in oracle[oracle.group!='pooled'].iterrows():
        lines.append(f'| {int(r.kappa)} | {r.group} | {r["mode"]} | {r.reference} | {r["mean"]:.6f} | {r.q05:.6f} | {r.q50:.6f} | {r.q95:.6f} |')
    lines.extend(['','[조건·집단·시드별 전체 비용](costs_by_parent.csv), [108개 모델 생성 평가](generation_metrics.csv), [120개 정답 생성 평가](oracle_metrics.csv), [검증](verification.json).'])
    (DOC/'result_tables.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    colors={'U':'#2466a3','G':'#c15d27'};shapes={'A':'o','B':'s','P':'^'}
    truth=oracle[(oracle.kappa==1)&(oracle.group=='1')&(oracle['mode']=='JOINT_CONT')&(oracle.reference=='raw')].iloc[0]
    axes[0].axhspan(truth.q05,truth.q95,color='gray',alpha=.14,label='Continuous oracle: 5–95% MC range')
    ordered=[(name,m) for name in ['U','G'] for m in ['A','B','P']]
    for i,(name,m) in enumerate(ordered):
        f=active[(active.name==name)&(active.variant==m)]
        mean=f.short_gap_repeat_curve_l1.mean();lo=f.short_gap_repeat_curve_l1.min();hi=f.short_gap_repeat_curve_l1.max()
        axes[0].errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt=shapes[m],color=colors[name],capsize=4,ms=8)
        axes[1].scatter(f.repeat_brier,f.short_gap_repeat_curve_l1,color=colors[name],marker=shapes[m],alpha=.25,s=35)
        axes[1].scatter(f.repeat_brier.mean(),mean,color=colors[name],marker=shapes[m],s=80)
        offset=(-32,-13) if (name,m)==('G','P') else ((5,10) if (name,m)==('G','B') else (5,4))
        axes[1].annotate(f'{name}/{m}',(f.repeat_brier.mean(),mean),xytext=offset,textcoords='offset points',fontsize=9)
    axes[0].set_xticks(range(6),['U/A','U/B','U/P','G/A','G/B','G/P'])
    axes[0].axhline(.03,color='#a22',ls='--',lw=1,label='Registered per-parent L1 limit')
    axes[0].set(ylabel='Generated repeat-curve L1',title='Active group: mean and training-seed range')
    axes[0].legend(fontsize=8,loc='best')
    axes[1].set(xlabel='Observed-history repeat Brier',ylabel='Generated repeat-curve L1',title='Prediction and generation (lower is better)')
    fig.savefig(DOC/'active_prediction_generation.png',dpi=180);plt.close(fig)
    curve=pd.read_csv(DOC/'repeat_curves.csv',dtype={'group':str})
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,constrained_layout=True)
    for ax,name in zip(axes,['U','G']):
        f=curve[(curve.kappa==1)&(curve.group=='1')&(curve.name==name)]
        for method,color in [('A','#667788'),('B','#3a8b63'),('P','#8556ad')]:
            v=f[f.variant==method].groupby('bin')[['reference','generated']].mean()
            ax.plot(range(1,6),v.generated,marker=shapes[method],label=method,color=color)
        ax.plot(range(1,6),v.reference,'k--',label='Validation reference')
        ax.set(title=f'{name}: active group',xlabel='Gap bin (short to long)',xticks=range(1,6));ax.legend()
    axes[0].set_ylabel('Observable repeat probability')
    fig.savefig(DOC/'active_repeat_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.3),constrained_layout=True)
    for i,(name,m) in enumerate([(n,v) for n in ['U','G'] for v in ['B','P']]):
        f=costs[(costs.name==name)&(costs.variant==m)&((costs.kappa==0)|(costs.group=='0'))].sort_values(['kappa','trial','group'])
        y=f.delta_fixed_history_range.to_numpy();assert len(y)==9
        ax.scatter(i+np.linspace(-.15,.15,len(y)),y,c=[colors[name] if v<=.01 else '#af2727' for v in y],s=45)
        ax.annotate(f'max {max(y):.4f}',(i,max(y)),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(.01,color='#af2727',ls='--',label='Registered increase limit: 0.01')
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xticks=range(4),xticklabels=['U/B','U/P','G/B','G/P'],ylim=(-.003,.03),
        ylabel='Added fixed-history repeat-probability range',title='Null groups: each dot is one condition / training seed')
    ax.legend(loc='upper left',fontsize=9);fig.savefig(DOC/'null_response_cost.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
