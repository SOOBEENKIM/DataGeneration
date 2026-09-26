"""Render completed audit tables without selecting favorable runs or filtering cells."""
import json
from run_argn_fraud_audit import DOCS, config, pd


def table(frame):
    def cell(x):
        if pd.isna(x): return 'NA'
        return f'{x:.6g}' if isinstance(x,float) else str(x)
    lines = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    cfg = config()
    m = pd.read_csv(DOCS/'generation_metrics.csv')
    assert len(m) == 4
    curves = pd.read_csv(DOCS/'position_curves.csv')
    noise = pd.read_csv(DOCS/'real_real_reference.csv')
    probe = pd.concat([pd.read_csv(DOCS/f'probe_seed_{s}.csv') for s in cfg['fit_seeds']])
    fits = pd.DataFrame([json.loads((DOCS/f'fit_{s}.json').read_text()) for s in cfg['fit_seeds']])
    risk = pd.read_csv(DOCS/'conditional_risk.csv')
    short = risk[risk.gap_bin.eq(1)].groupby('source')[['events','frauds','invalid_labels']].sum()
    short = short.reindex(['original_validation', *m.run],fill_value=0)
    short['fraud_rate'] = short.frauds / short.events.where(short.events.gt(0))
    short.reset_index().to_csv(DOCS/'positive_gap_at_most_5s.csv',index=False)
    relation_cols = ['merchant_category_tv','category_amount_tv','gap_previous_category_category_tv',
                    'history_amount_tv','class_0_gap_previous_category_category_tv',
                    'class_1_gap_previous_category_category_tv','class_1_category_amount_tv','class_1_history_amount_tv']
    reference = pd.DataFrame({c:[noise[c].min(),noise[c].median(),noise[c].max()] for c in relation_cols},index=['min','median','max'])
    report = '\n\n'.join([
        '# 라벨 포함 공식 ARGN: 전체 결과표',
        '2026-09-27. 두 학습 seed × 두 생성 seed. TV는 낮을수록 좋다. 모든 수치는 재사용된 development validation 결과다. 범위는 신뢰구간이 아니다.',
        '## 실제 학습량', table(fits[['seed','last_epoch','selected_epoch','updates','seconds','selected_check_loss','budget_reached']]),
        '## 생성 수와 라벨',table(m[['run','real_events','generated_events','real_fraud_rate','generated_fraud_rate','generated_frauds','invalid_generated_label_rate','invalid_amount_rate','invalid_gap_rate']]),
        '## 관계 TV',table(m[['run',*relation_cols]]),
        '## 실제 고객을 반으로 나눈 real–real 참고 범위',
        '12회 분할. 비교 표본 크기와 고객 구성이 생성 비교와 다르므로 통계적 기준선/유의성 검정으로 사용하지 않는다.',table(reference.reset_index(names='summary')),
        '## 첫 거래의 fit에 없던 가맹점–유형 조합',
        table(curves[curves.position_band.eq(0)][['run','real_events','generated_events','real_pair_absent_from_fit','generated_pair_absent_from_fit']]),
        '## 실제 이력에서의 거래 유형 예측',
        '각 고객의 최초512개 거래까지, 총69,086개. 현재 merchant embedding만 교란하며 이전 이력은 그대로 둔다. 조회표 진단에는 같은 prefix와 전체177,997개 결과를 모두 기록한다.',
        table(probe[(probe.field.eq('category')) & probe.group.eq('all')][['run','intervention','events','nll','accuracy']]),
        '## 0초 초과 5초 이하 간격',
        '0건의 조건은 사기율 NA다. 실제 표본45개·사기1개인 조건의 일치/불일치만으로 우월성을 결론내리지 않는다.',table(short.reset_index()),
        '상세 조건별 빈도·사기율은 conditional_risk.csv, 위치별 고객/거래 수는 position_curves.csv, 정상·사기별 실제 이력 예측은 probe_seed_*.csv에 남긴다.'
    ])+'\n'
    (DOCS/'result_tables.md').write_text(report)
    print(table(m[['run','generated_events','generated_fraud_rate','merchant_category_tv','class_1_gap_previous_category_category_tv']]))


if __name__ == '__main__': main()
