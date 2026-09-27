"""Independent raw-output audit and registered screens for temporal density."""
import argparse
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from run_argn_time_density import ROOT,OUT,DOCS,CFG,verify,digest,write
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate,summary,generation_path,w1
from report_argn_joint_preservation import independent_numbers

ARMS=['frozen','boundary_fit','phase_mixture']+CFG['arms']
DISPLAY=dict(frozen='고정 개선본',boundary_fit='기존 경계 head',phase_mixture='단순 phase GMM',
    history_only='일반 이력 밀도',history_clock='이력 + 개인 속도 입력',history_relative='개인 속도 대비 출력')


def markdown_table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    lines += ['| '+' | '.join(str(v).replace('|','\\|') for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def locate(arm,fs,gs):
    fresh=gs>=20261021
    if arm=='frozen':
        p=generation_path('onset_fit',fs,gs)
        doc=ROOT/'docs/argn_joint_preservation_v1/onset_output'
        if fresh:doc=doc/'confirmation'
        metric=doc/f'evaluation/metrics_onset_fit_{fs}.csv'
    elif arm in ['boundary_fit','phase_mixture']:
        group='argn_boundary_gap_v1' if arm=='boundary_fit' and not fresh else 'argn_phase_gap_v1'
        suffix='/confirmation' if fresh else ''
        p=ROOT/f'artifacts/{group}{suffix}/runs/{arm}_{fs}/generated_validation_{gs}.parquet'
        metric=ROOT/f'docs/{group}{suffix}/evaluation/metrics_{arm}_{fs}.csv'
    else:
        p=OUT/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet'
        metric=DOCS/f'evaluation/metrics_{arm}_{fs}.csv'
    return p,metric


def collect(arms):
    real=pd.read_parquet(SOURCE/'prepared/validation.parquet');rd=decorate(real)
    rows=[];phases=[];personal=[];checks=[];sampling=[];total=0
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            baseline=pd.read_parquet(generation_path('onset_fit',fs,gs));cols=['entity_id','event_index','event_is_fraud']
            for arm in arms:
                p,metric=locate(arm,fs,gs);assert (p.parent/'COMPLETE.json').exists(),p
                g=json.loads((p.parent/f'GENERATION_{gs}.json').read_text());assert digest(p)==g['sha256']
                raw=pd.read_parquet(p);assert raw[cols].equals(baseline[cols]);info=dict(arm=arm,fit_seed=fs,generation_seed=gs)
                m=pd.read_csv(metric);m=m[m.generation_seed.eq(gs)].iloc[0].to_dict();m.update(info)
                nums=independent_numbers(real,raw)
                for k in ['normal_amount_log_w1','fraud_amount_log_w1','normal_gap_seconds_log_w1','fraud_gap_seconds_log_w1','fraud_ratio_log_w1']:
                    assert abs(m[k]-nums[k])<1e-12,(info,k,m[k],nums[k])
                checks.append(dict(**info,rows=len(raw),**nums,exact_labels_lengths=True,output_sha256=g['sha256']))
                sd=decorate(raw);ph=summary(rd,sd,info);phases+=ph
                for phase in ['onset','return_normal','normal_stay','continuation','left_fraud']:
                    a=rd[rd.phase.eq(phase)&rd.past_gap.gt(0)];b=sd[sd.phase.eq(phase)&sd.past_gap.gt(0)]
                    value=w1(a.gap/a.past_gap,b.gap/b.past_gap)
                    personal.append(dict(**info,phase=phase,real_events=len(a),generated_events=len(b),
                        real_customers=a.entity_id.nunique(),generated_customers=b.entity_id.nunique(),relative_gap_w1=value))
                    m[f'{phase}_personal_gap_w1']=value
                for phase in ['onset','return_normal']:
                    m[f'{phase}_gap_w1']=next(x['log_w1'] for x in ph if x['group']==phase and x['field']=='gap')
                rows.append(m)
                if arm in CFG['arms']:
                    total+=len(raw);h=OUT/f'worker_{fs}/{arm}/time_head.pt';assert digest(h)==g['head_sha256']
                audit=g.get('sampling_audit',[])
                sampling.append(dict(**info,clipped_values=sum(x['clipped_values'] for x in audit),sampled_values=sum(x['sampled_values'] for x in audit)))
                print('DENSITY_AUDITED',arm,fs,gs,flush=True)
    return pd.DataFrame(rows),pd.DataFrame(phases),pd.DataFrame(personal),pd.DataFrame(checks),pd.DataFrame(sampling),total


def screen(data):
    means=data.groupby(['fit_seed','arm']).mean(numeric_only=True);rows=[]
    for fs in CFG['fit_seeds']:
        b=means.loc[(fs,'frozen')];simple=means.loc[(fs,'phase_mixture')];history=means.loc[(fs,'history_clock')]
        for arm in CFG['arms']:
            a=means.loc[(fs,arm)];cost=[]
            for k,delta in [('fraud_ratio_log_w1',.02),('normal_amount_log_w1',.02),('normal_gap_seconds_log_w1',.02),
                ('fraud_gap_seconds_log_w1',.05),('class_0_merchant_tv',.02),('normal_stay_personal_gap_w1',.02)]:
                if not np.isfinite(a[k]) or a[k]>b[k]+delta:cost.append('frozen:'+k)
            if a.median_unique_merchants_per_customer<b.median_unique_merchants_per_customer*.95:cost.append('frozen:merchant_diversity')
            for k in ['fraud_gap_seconds_log_w1','normal_gap_seconds_log_w1']:
                if a[k]>simple[k]+.03:cost.append('mixture:'+k)
            primary=bool(a.onset_personal_gap_w1<simple.onset_personal_gap_w1)
            extra=bool(a.onset_personal_gap_w1<history.onset_personal_gap_w1)
            rows.append(dict(fit_seed=fs,arm=arm,onset_better_than_simple=primary,onset_better_than_clock=extra,
                costs=','.join(cost),passes_quality_screen=primary and not cost,
                supports_relative_added_effect=arm=='history_relative' and primary and extra and not cost))
    return pd.DataFrame(rows)


def plot(data,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
    colors=dict(zip(ARMS,['#7f7f7f','#b44d5e','#db9c29','#5093b1','#34836e','#694ca4']))
    labels=dict(frozen='Frozen candidate',boundary_fit='Boundary head',phase_mixture='Phase GMM',history_only='History density',
        history_clock='History + clock',history_relative='Clock-relative density')
    means=data.groupby(['arm','fit_seed']).mean(numeric_only=True)
    for arm in ARMS:
        if arm not in data.arm.unique():continue
        d=means.loc[arm]
        for ax,x in zip(axes,['fraud_gap_seconds_log_w1','normal_stay_personal_gap_w1']):
            ax.scatter(d[x],d.onset_personal_gap_w1,color=colors[arm],alpha=.55,s=38)
            ax.scatter(d[x].mean(),d.onset_personal_gap_w1.mean(),color=colors[arm],marker='X',s=130,label=labels[arm])
    axes[0].set_xlabel('All fraud gap log-W1 (lower is better)');axes[1].set_xlabel('Normal personal-gap W1 (lower is better)')
    for ax in axes:
        ax.set_ylabel('Fraud-onset personal-gap W1 (lower is better)');ax.grid(alpha=.18)
    axes[1].legend(fontsize=8)
    fig.suptitle('Single time-output study: 2 frozen parent fits, 4 draws per fit')
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'time_density_tradeoff.{ext}',dpi=180)
    plt.close(fig)


def main(controls_only=False):
    verify();arms=ARMS[:3] if controls_only else ARMS
    dest=DOCS/'controls' if controls_only else DOCS;dest.mkdir(exist_ok=True,parents=True)
    cached=DOCS/'controls/independent_checks.csv'
    if not controls_only and cached.exists():
        # Reuse the already independently audited controls; recheck every raw hash.
        old_checks=pd.read_csv(cached)
        assert len(old_checks)==24
        for row in old_checks.itertuples():
            p,_=locate(row.arm,row.fit_seed,row.generation_seed)
            assert digest(p)==row.output_sha256
        new_parts=[];total=0
        for arm in CFG['arms']:
            cache=DOCS/f'arm_audits/{arm}'
            names=['all_metrics','phase_metrics','personal_gap','independent_checks','sampling_audit']
            if all((cache/f'{name}.csv').exists() for name in names):
                frames=[pd.read_csv(cache/f'{name}.csv') for name in names]
                assert len(frames[3])==8 and set(frames[3].arm)=={arm}
                for row in frames[3].itertuples():
                    p,_=locate(row.arm,row.fit_seed,row.generation_seed);assert digest(p)==row.output_sha256
                count=int(frames[3].rows.sum())
            else:
                result=collect([arm]);frames=result[:5];count=result[-1]
            new_parts.append(frames);total+=count
        data,phases,personal,checks,sampling=[pd.concat([part[i] for part in new_parts],ignore_index=True) for i in range(5)]
        merged=[]
        for name,frame in [('all_metrics',data),('phase_metrics',phases),('personal_gap',personal),('independent_checks',checks),('sampling_audit',sampling)]:
            merged.append(pd.concat([pd.read_csv(DOCS/f'controls/{name}.csv'),frame],ignore_index=True))
        data,phases,personal,checks,sampling=merged
    else:
        data,phases,personal,checks,sampling,total=collect(arms)
    for name,frame in [('all_metrics',data),('phase_metrics',phases),('personal_gap',personal),('independent_checks',checks),('sampling_audit',sampling)]:frame.to_csv(dest/f'{name}.csv',index=False)
    means=data.groupby(['arm','fit_seed']).mean(numeric_only=True);means.to_csv(dest/'fit_means.csv')
    keys=['onset_personal_gap_w1','onset_gap_w1','fraud_gap_seconds_log_w1','normal_gap_seconds_log_w1',
        'normal_stay_personal_gap_w1','fraud_ratio_log_w1','normal_amount_log_w1','class_0_merchant_tv','median_unique_merchants_per_customer']
    data.groupby('arm')[keys].agg(['mean','min','max']).to_csv(dest/'ranges.csv')
    lines=['# 단일 조건부 시간 출력 비교','',
        '같은 GPU의2부모×4생성 평균. W1은 작을수록 좋다. CPU 대조는 섞지 않았다.',
        '새 checkpoint 선택은 내부 검증만 사용했다. 이 표는 development이며 최종 test는 미사용이다.','',
        '| 모델 | 개인 대비 시작 간격 | 시작 간격 | 전체 사기 간격 | 정상 간격 | 정상 개인 간격 | 개인 대비 사기 금액 | 정상 금액 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        m=data[data.arm.eq(arm)].mean(numeric_only=True)
        lines.append('| '+DISPLAY[arm]+' | '+' | '.join(f'{m[k]:.4f}' for k in keys[:7])+' |')
    if not controls_only:
        gates=screen(data);gates.to_csv(dest/'screen.csv',index=False)
        lines+=['','## 사전 screen','',markdown_table(gates),'',
            'supports_relative_added_effect는 두 부모 모두 통과해야 한다. 생성4회는 독립학습4회가 아니다.',
            '전체 지표·범위·부모별 값·변경 비용은 CSV에 보존한다. 최신 생성기 전체 직접 비교는 이번 실행 범위 밖이며 아직 완료하지 않았다.','']
        fitrows=[]
        for fs in CFG['fit_seeds']:
            for arm in CFG['arms']:
                p=OUT/f'worker_{fs}/{arm}';f=json.loads((p/'FIT_COMPLETE.json').read_text());assert digest(p/'time_head.pt')==f['sha256']
                route=json.loads((p/'ROUTING_CHECK.json').read_text());assert route['strict_past_clock_exact'] and route['strict_native_checkpoint']
                fitrows.append(f)
                pd.read_csv(p/'learning_curve.csv').to_csv(dest/f'fits/learning_{arm}_{fs}.csv',index=False)
                write(dest/f'fits/routing_{arm}_{fs}.json',route)
        pd.DataFrame(fitrows).to_csv(dest/'fits.csv',index=False)
        conditional=pd.concat([pd.read_csv(DOCS/f'conditional_{fs}.csv') for fs in CFG['fit_seeds']],ignore_index=True)
        conditional.to_csv(dest/'conditional_all.csv',index=False)
        conditional.groupby(['arm','split','phase'])[['gap_w1','personal_gap_w1','macro_nll']].mean().to_csv(dest/'conditional_means.csv')
        plot(data,dest)
        write(dest/'COMPLETE.json',dict(new_small_fits=6,new_full_generations=24,new_generated_rows=total,
            audited_full_generations=len(checks),same_label_length_comparisons=len(checks),test_events_read=False,
            source_sha256=digest(__file__),completed_utc=datetime.now(timezone.utc).isoformat()))
    (dest/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controls-only',action='store_true');a=p.parse_args();main(a.controls_only)
