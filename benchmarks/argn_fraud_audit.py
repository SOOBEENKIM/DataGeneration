"""Direct transaction-relation diagnostics; labels are outcomes, never future context."""
import numpy as np
import pandas as pd

MERCHANT = 'receiver_or_mark'
AMOUNT = 'amount_or_numeric_value'
LABEL = 'event_is_fraud'
CORE = ['gap', MERCHANT, AMOUNT, 'category', LABEL]
STATIC = ['cardholder_gender', 'cardholder_state', 'birth_year', 'city_population']


def tv(a, b, cols):
    if not len(a) or not len(b):
        return None
    x = a[cols].value_counts(normalize=True, dropna=False)
    y = b[cols].value_counts(normalize=True, dropna=False)
    return float(x.subtract(y, fill_value=0).abs().sum() / 2)


def metric_state(fit, config):
    return dict(amount_edges=np.unique(np.quantile(fit[AMOUNT], [.1,.25,.5,.75,.9,.95,.99])).tolist(),
                gap_edges=config['gap_edges_seconds'], position_edges=config['position_edges'])


def features(frame, state):
    d = frame.sort_values(['entity_id', 'event_index'], kind='stable').copy().reset_index(drop=True)
    # Native parquet uses pandas nullable numeric dtypes; normalize missing values to numpy NaN.
    for c in ['gap', AMOUNT]:
        d[c] = pd.to_numeric(d[c], errors='raise').astype(float)
    if d.duplicated(['entity_id', 'event_index']).any():
        raise ValueError('Duplicated customer/event position')
    d['gap_bin'] = np.searchsorted(state['gap_edges'], d.gap.fillna(-1).to_numpy(), side='left')
    d.loc[d.event_index.eq(0), 'gap_bin'] = -1
    d['amount_bin'] = np.searchsorted(state['amount_edges'], d[AMOUNT].to_numpy(), side='left')
    d['position_band'] = np.searchsorted(state['position_edges'], d.event_index.to_numpy(), side='left')
    d['previous_category'] = d.groupby('entity_id', sort=False).category.shift().fillna('<START>')
    d['seen_merchant'] = d.duplicated(['entity_id', MERCHANT]).astype(int)
    prior = d.groupby('entity_id', sort=False)[AMOUNT].transform(lambda x: x.shift().rolling(20, min_periods=5).median())
    ratio = d[AMOUNT] / prior.where(prior.gt(0))
    d['amount_vs_history'] = np.where(ratio.isna(), -1, np.where(ratio <= 1, 0, np.where(ratio <= 3, 1, 2)))
    if LABEL in d:
        label = pd.to_numeric(d[LABEL], errors='coerce')
        d['fraud'] = label.where(label.isin([0, 1]), -1).astype(int)
    return d


def summaries(real, synthetic, state):
    r, s = features(real, state), features(synthetic, state)
    result = dict(real_events=len(r), generated_events=len(s), real_customers=r.entity_id.nunique(),
                  generated_customers=s.entity_id.nunique())
    result['invalid_amount_rate'] = float((~np.isfinite(s[AMOUNT]) | s[AMOUNT].lt(0)).mean())
    gap = s.loc[s.event_index.gt(0), 'gap']
    result['invalid_gap_rate'] = float((~np.isfinite(gap) | gap.lt(0)).mean()) if len(gap) else None
    pairs = {
        'merchant_category_tv': [MERCHANT, 'category'],
        'category_amount_tv': ['category', 'amount_bin'],
        'gap_previous_category_category_tv': ['gap_bin', 'previous_category', 'category'],
        'history_amount_tv': ['gap_bin', 'seen_merchant', 'amount_vs_history'],
        'category_tv': ['category'], 'merchant_tv': [MERCHANT], 'amount_tv': ['amount_bin'], 'gap_tv': ['gap_bin']}
    for name, cols in pairs.items():
        rr, ss = (r[r.event_index.gt(0)], s[s.event_index.gt(0)]) if name.startswith('gap') else (r, s)
        result[name] = tv(rr, ss, cols)
    if 'fraud' in r and 'fraud' in s:
        result.update(real_fraud_rate=float(r.fraud.eq(1).mean()), generated_fraud_rate=float(s.fraud.eq(1).mean()),
                      real_frauds=int(r.fraud.eq(1).sum()), generated_frauds=int(s.fraud.eq(1).sum()),
                      invalid_generated_label_rate=float(s.fraud.eq(-1).mean()),
                      label_tv=tv(r, s, ['fraud']),
                      history_amount_label_tv=tv(r, s, ['gap_bin','seen_merchant','amount_vs_history','fraud']))
        for label in [0, 1]:
            for name in ['category_amount_tv', 'gap_previous_category_category_tv', 'history_amount_tv']:
                result[f'class_{label}_{name}'] = tv(r[(r.fraud == label) & r.event_index.gt(0)],
                                                     s[(s.fraud == label) & s.event_index.gt(0)], pairs[name])
    return result


def position_curves(real, synthetic, fit, state):
    r, s = features(real, state), features(synthetic, state)
    supported = pd.MultiIndex.from_frame(fit[[MERCHANT, 'category']].drop_duplicates())
    for d in [r, s]:
        d['unseen_pair'] = ~pd.MultiIndex.from_frame(d[[MERCHANT, 'category']]).isin(supported)
    rows = []
    for band in sorted(set(r.position_band) | set(s.position_band)):
        rr, ss = r[r.position_band.eq(band)], s[s.position_band.eq(band)]
        row = dict(position_band=int(band), real_events=len(rr), generated_events=len(ss),
                   real_customers=rr.entity_id.nunique(), generated_customers=ss.entity_id.nunique(),
                   merchant_category_tv=tv(rr, ss, [MERCHANT, 'category']),
                   real_pair_absent_from_fit=float(rr.unseen_pair.mean()) if len(rr) else None,
                   generated_pair_absent_from_fit=float(ss.unseen_pair.mean()) if len(ss) else None)
        for name, d in [('real', rr), ('generated', ss)]:
            if 'fraud' in d:
                row[name+'_frauds'] = int(d.fraud.eq(1).sum())
                row[name+'_fraud_rate'] = float(d.fraud.eq(1).mean()) if len(d) else None
        rows.append(row)
    return rows


def risk_table(frame, state, source):
    d = features(frame, state)
    cols = ['gap_bin','seen_merchant','amount_vs_history']
    out = d.groupby(cols, dropna=False).agg(events=('fraud','size'), customers=('entity_id','nunique'),
        frauds=('fraud',lambda x: x.eq(1).sum()), invalid_labels=('fraud',lambda x: x.eq(-1).sum())).reset_index()
    out['fraud_rate'] = out.frauds / out.events
    out['event_share'] = out.events / len(d)
    out['source'] = source
    return out
