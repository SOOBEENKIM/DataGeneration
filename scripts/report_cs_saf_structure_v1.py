"""Descriptive tables and standalone figures; no model or threshold selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/cs_saf/structure_v1'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    summary=json.loads((DOC/'execution_summary.json').read_text());assert summary['status']=='COMPLETE'
    c=pd.read_csv(DOC/'conditional_metrics.csv',dtype={'group':str})
    g=pd.read_csv(DOC/'generation_by_parent.csv',dtype={'group':str})
    decision=json.loads((DOC/'decision.json').read_text())
    variants=['raw','gap','direct'];names=['U','G','C'];colors=['#18836d','#95688b','#3b6ea8']
    rows=[]
    for name in names:
        for variant in variants:
            cr=c[(c.kappa==1)&(c.group=='1')&(c.name==name)&(c.variant==variant)]
            gr=g[(g.kappa==1)&(g.group=='1')&(g.name==name)&(g.variant==variant)]
            rows.append([name,variant,f'{cr.repeat_brier.mean():.6f}',f'{cr.mark_nll.mean():.6f}',f'{cr.curve_l1.mean():.6f}',
                f'{gr.short_gap_repeat_curve_l1.mean():.6f}',
                '['+', '.join(f'{v:.6f}' for v in gr.sort_values('trial').short_gap_repeat_curve_l1)+']'])
    text='# 전체 결과 표\n\n관계가 있는 10% 집단. 세 학습 시드의 평균이며 각 생성 값은 부모별 생성 난수 3개를 먼저 평균했다.\n\n'
    text+=table(['구조','보정','거래별 반복 Brier','행동 NLL','실제 이력 곡선 L1','생성 곡선 L1','학습 시드별 생성 L1'],rows)
    text+='\n\n## 사전등록 1차 비교\n\n'
    text+=table(['후보','대조군','평균 절대 감소','평균 상대 감소','개선 시드','생성 개선 screen'],
        [[r['candidate'],r['comparator'],f"{r['absolute_mean_reduction']:.6f}",f"{r['relative_mean_reduction']*100:.2f}%",
          f"{r['improving_seeds']}/3",r['passes_generation_improvement']] for r in decision['generation_contrasts'] if r['role']=='primary'])
    text+='\n\n'+table(['판정','통과'],[[k,decision[k]] for k in ['basic_prediction_all_pass','conditional_cost_all_pass',
        'distribution_cost_all_pass','null_all_pass','primary_pass']])
    text+='\n\n보정 후 C의 비교는 보조 분석이며 보정 전 구조의 실패를 대체하지 않는다. 전체 실패 cell은 primary_costs.csv에 보존한다.\n'
    (DOC/'result_tables.md').write_text(text)

    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained')
    for ax,frame,metric,title,ylabel in [
        (axes[0],g,'short_gap_repeat_curve_l1','A  Freely generated repeat relation','Repeat-curve L1 (lower is better)'),
        (axes[1],c,'repeat_brier','B  Prediction on real histories','Repeat Brier score (lower is better)')]:
        active=frame[(frame.kappa==1)&(frame.group=='1')]
        for name,color,offset in zip(names,colors,[-.08,0,.08]):
            stat=active[active.name==name].groupby('variant')[metric].agg(['mean','min','max']).loc[variants]
            mean=stat['mean'].to_numpy();ax.errorbar(np.arange(3)+offset,mean,
                yerr=np.stack([mean-stat['min'].to_numpy(),stat['max'].to_numpy()-mean]),
                marker='o',color=color,label=name,linewidth=1.8,capsize=4)
        ax.set(title=title,ylabel=ylabel,xticks=np.arange(3),xticklabels=['Raw','Gap calibration','Direct repeat'])
        ax.grid(axis='y',alpha=.2)
    axes[0].axhline(.03,ls='--',color='#777',lw=1,label='Screen: .03');axes[0].legend(frameon=False)
    fig.suptitle('Matched structure pilot: U / general observed repeat G / constrained C',fontsize=15,fontweight='bold')
    fig.supxlabel('133,549 parameters each; 3 paired training seeds. Bars: seed range, not confidence intervals.\n'
        'Active 10% group; 3 generation tapes averaged within a trained model. Existing synthetic data; exploratory.',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(DOC/f'active_comparison.{ext}',bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    primary=[r for r in decision['generation_contrasts'] if r['role']=='primary']
    for i,r in enumerate(primary):
        axes[0].scatter(np.repeat(i,3)+np.linspace(-.08,.08,3),r['paired_differences'],color=colors[2],s=45)
        axes[0].scatter(i,np.mean(r['paired_differences']),marker='_',s=250,color='black')
    axes[0].axhline(0,color='#666',lw=1);axes[0].axhline(-.002,ls='--',color='#888',lw=1)
    axes[0].set(title='A  Active generation: paired differences',xticks=[0,1],
        xticklabels=['C raw - G raw','C raw - U gap'],ylabel='Repeat-curve L1 difference (negative favors C)')
    costs=pd.read_csv(DOC/'primary_costs.csv',dtype={'group':str})
    active=costs[(costs.kappa==1)&(costs.group=='1')]
    for i,name in enumerate(['G/raw','U/gap']):
        values=active[active.comparator==name].sort_values('trial').delta_repeat_brier.to_numpy()
        axes[1].scatter(np.repeat(i,3)+np.linspace(-.08,.08,3),values,color=colors[2],s=45)
    axes[1].axhline(0,color='#666',lw=1);axes[1].axhline(.002,ls='--',color='#b64646',lw=1)
    axes[1].set(title='B  Active prediction cost',xticks=[0,1],xticklabels=['C raw - G raw','C raw - U gap'],
        ylabel='Brier difference (positive is a cost)')
    fig.supxlabel('Dots: 3 paired training seeds. Dashed lines: absolute generation target and maximum Brier cost.\n'
        'These panels show only active cells; the registered decision also checks all null groups and other costs.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(DOC/f'primary_pairs.{ext}',bbox_inches='tight')
    plt.close(fig)
    print(text)


if __name__=='__main__':main()
