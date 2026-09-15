# AMLSim and Sparkov external-validation feasibility audit

## Decision

Both datasets are technically usable for a separately preregistered external sequence study, but this audit does **not** authorize an experiment. AMLSim is `FEASIBLE_WITH_PREREGISTRATION`; Sparkov additionally needs a decision about the published test split because card entities overlap. CoF-SeqGen v3 is closed and is not the external baseline.

All reported statistics came from read-only CSV inspection. GPU/CUDA, model fit/sample, data generation, validation/test execution, TSTR, privacy, and full runs were all zero.

## Frozen existing CoF baseline (non-v3)

- Source commit: `99a445f6dc893a8c2240d950de4f92877cc07f8a`
- Model: `models/cof_seqgen.py`, SHA-256 `caf5fdb3baf367a28c6081a8ba5f4a89e59dacbba2377636acba5c051ae7006e`
- Adapter: `generators/cof_seqgen_adapter.py`, SHA-256 `48e9e1280bc23a27abf0b5f6447f6682e3b846d0d1773fb96d99abcf407dd7f3`
- Config: `configs/benchmark_v2/full_v2_5.yaml`, SHA-256 `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`
- Identity: existing CoF-SeqGen v2.5, not v3. The exact frozen model block is 20,000 requested updates, 7,200 s cap, d_model 128, 2 layers, batch 256, Adam 1e-3, diffusion steps 50, and coherence_lambda 0.0.
- This freezes identity only. No external preprocessing, execution budget, fit, sample, or test is authorized.

## AMLSim: observed raw schema and distributions

Raw transaction file: `<LOCAL_DATA_ROOT>/ibm_amlsim_example_accounts_transactions_alerts/transactions.csv`; SHA-256 `cc4885da00fd4c6f854aae4e7bc8dd9c35a9b491f5ef871c51c9194ee1e292f7`; 1,323,234 rows.

Actual transaction columns: `TX_ID`, `SENDER_ACCOUNT_ID`, `RECEIVER_ACCOUNT_ID`, `TX_TYPE`, `TX_AMOUNT`, `TIMESTAMP`, `IS_FRAUD`, `ALERT_ID`.

| Field | Actual column | Audit choice |
|---|---|---|
| Timestamp | `TIMESTAMP` | Coarse integer simulation step |
| Entity | `SENDER_ACCOUNT_ID` | Recommended sequence owner |
| Receiver | `RECEIVER_ACCOUNT_ID` | Recommended counterparty |
| Amount | `TX_AMOUNT` | Train-only `log1p` then standardization is feasible |
| Row fraud | `IS_FRAUD` | Never an input; window Y is any-fraud |
| Stable tie-breaker | `TX_ID` | Required because timestamps repeat |

- Entities: 9,999; receivers: 9,926.
- Transaction fraud: 1,719/1,323,234 (0.129909%).
- Entity any-fraud: 1,484/9,999 (14.841484%).
- Non-overlapping 32-event window any-fraud: 1,611/45,501 (3.540582%).
- Non-overlapping 10-step window any-fraud: 1,697/159,659 (1.062890%).
- Within-entity duplicate-timestamp rows: 888,787; duplicate entity-time groups: 84,283.
- Missing protocol fields: `{"IS_FRAUD": 0, "RECEIVER_ACCOUNT_ID": 0, "SENDER_ACCOUNT_ID": 0, "TIMESTAMP": 0, "TX_AMOUNT": 0, "TX_ID": 0}`; duplicate transaction-ID rows: 0.

| AMLSim entity transactions | min | p05 | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Count | 7 | 16 | 132 | 340 | 740 | 963 |

The proposed entity split has 6,979/1,518/1,502 train/validation/internal-test senders. A train-only receiver vocabulary has 9,686 categories; validation OOV is 3,244 rows (1.663598%) and internal-test OOV is 3,695 rows (1.929433%).

Recommended mapping: sender account entity, stable event order `(SENDER_ACCOUNT_ID, TIMESTAMP, TX_ID)`, non-overlapping 32-event windows with masked tails, receiver account, and `Y=1` iff any row in the window has `IS_FRAUD=1`. Split sender entities by a frozen hash. `ALERT_ID`, `IS_FRAUD`, and `TX_ID` are excluded from model features. Receiver OOV is handled by a train-fitted vocabulary and explicit UNK.

## Sparkov: observed raw schema and distributions

Train file: `<LOCAL_DATA_ROOT>/Sparkov_fraud_train_test/fraudTrain.csv`; SHA-256 `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d`; 1,296,675 rows. Locked published test: `<LOCAL_DATA_ROOT>/Sparkov_fraud_train_test/fraudTest.csv`; SHA-256 `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0`; 555,719 rows.

Actual columns: `Unnamed: 0`, `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `first`, `last`, `gender`, `street`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`, `is_fraud`. The first empty CSV header is an exported index and is not a feature.

| Field | Actual column | Audit choice |
|---|---|---|
| Timestamp | `unix_time` (`trans_date_trans_time` is human-readable) | Stable numeric order |
| Entity | `cc_num` | Recommended card sequence owner |
| Receiver | `merchant` | Recommended counterparty identity |
| Secondary receiver diagnostic | `category` | Merchant type, not receiver identity |
| Amount | `amt` | Train-only `log1p` then standardization is feasible |
| Row fraud | `is_fraud` | Never an input; window Y is any-fraud |
| Stable tie-breaker | `trans_num` | Deterministic tie-breaker |

- Train entities: 983; merchants: 693; categories: 14.
- Train transaction fraud: 7,506/1,296,675 (0.578865%).
- Train entity any-fraud: 762/983 (77.517803%).
- Train 32-event window any-fraud: 946/41,019 (2.306248%).
- Locked-test transaction fraud: 2,145/555,719 (0.385986%).
- Published test entities overlapping train: 908/924 (98.268398%).
- Missing train protocol fields: `{"amt": 0, "category": 0, "cc_num": 0, "is_fraud": 0, "merchant": 0, "trans_num": 0, "unix_time": 0}`; duplicate transaction-ID rows: 0.

| Sparkov train entity transactions | min | p05 | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Count | 7 | 10.1 | 1,054.0 | 2,921.6 | 3,099.4 | 3,123 |

The fraudTrain-only entity split has 684/143/156 train/validation/internal-test cards. All 693 merchants appear in the proposed train split, giving zero merchant OOV rows in those feasibility partitions; this observed zero must not remove the required UNK contract. The published test also has zero merchant OOV rows relative to fraudTrain, but 98.27% entity overlap remains the dominant split concern.

Recommended mapping: card entity, stable `(cc_num, unix_time, trans_num)` order, non-overlapping 32-event windows with masked tails, `merchant` receiver, and window any-fraud Y. Development uses only `fraudTrain.csv` with an entity-disjoint hash split. `fraudTest.csv` remains untouched until a separately preregistered final temporal robustness evaluation; it is not entity-disjoint and must never calibrate transforms, vocabulary, thresholds, or candidates.

## Train-only transforms, vocabulary, and leakage control

For both datasets, fit amount `log1p`/standardization, time or gap bins, receiver vocabulary/UNK, sequence length/tail policy, and any external hard thresholds on the training entities only. Freeze source/data/config hashes before validation. Validation selects only a preregistered protocol; the locked test is read once. Entity IDs themselves are not model features.

The proposed deterministic feasibility split is SHA-256(entity) modulo 100: 70% train, 15% validation, 15% internal test. The exact split counts and receiver OOV rates are in the JSON evidence. This is a feasibility calculation, not a final preregistration.

## Existing preparation-code compatibility

`data/prepare_amlsim.py` (SHA-256 `381ea3482f64c2a3bd686adea8e5c569c9c8754cf48fe0831f89716380ba77a7`) agrees with the raw sender/receiver/amount/timestamp mapping, but its final order omits `TX_ID`; it is not admissible unchanged because 888,787 rows participate in within-sender timestamp ties. `data/prepare_sparkov.py` (SHA-256 `2b4ed307c2137433a9df6d4d836a052ca74d6148b7ef6e6071c3f52e5a7c630e`) concatenates supplied files and drops `merchant` in favor of `category`; it is not admissible because that can mix the published test into preparation and discards the actual counterparty identity.

## Row-shuffle negative control

Within every entity window, keep timestamp positions, Y, length, padding mask, and row-feature marginals fixed; independently permute aligned non-time feature tuples such as `(amount, receiver)` over valid positions. Recompute gap-aligned summaries without exposing row-level fraud labels. Compare the same order-sensitive statistic/classifier on original and shuffled sequences. If discrimination or coherence does not fall, the external signal is primarily row-level and a sequence claim is unsupported. Orientation, thresholding, and selection are train/validation-only.

## External fidelity/coherence protocol (proposal only)

The controlled benchmark's five metric families remain useful diagnostics, but its numeric thresholds must **not** be transplanted. External hard thresholds require train-only entity bootstrap and preregistration. Report row fidelity (amount/gap marginals and Y effects, receiver PMF/UNK, length/Y prevalence, support/mask) separately from sequence coherence (repeat/run/transition, joint gap-receiver, cross-channel dependence, original-versus-shuffled delta). The locked test cannot revise either family.

## Risks, exclusion criteria, and decisions still required

- Approve non-overlapping 32-event windows (with masked tails) versus a separately preregistered time-window alternative.
- Approve Sparkov merchant as primary receiver and category as secondary diagnostic.
- Choose whether Sparkov fraudTest.csv is a final temporal robustness evaluation despite entity overlap, or whether only the fraudTrain-derived entity-disjoint internal test is confirmatory.
- Approve train-only external-data thresholds and the row-shuffle coherence endpoint before any model execution.
- Approve the frozen v2.5 non-v3 CoF source/config identity and a separate external execution budget; neither is authorized by this audit.

Dataset-specific risks and exclusion criteria are recorded verbatim in the JSON evidence. If any required protocol field is missing, the transaction ID cannot deterministically break timestamp ties, a frozen split lacks positive validation/test windows, or Sparkov's published test is used for calibration, execution must be refused.

## Preservation and execution accounting

- Audit base HEAD: `91acaf97fee2ae6ada305670c06698b98a247561`
- Existing v2.5-v3 runtime, data, configs, checkpoints, and forensic artifacts were read only and were not regenerated or moved.
- Preservation anchors reverified: v2.5 `FINAL_COMPLETE.json` SHA-256 `e47d46b9...57d8a`, frozen-data manifest `b2529f00...3e05`, v2.8 candidate tree `7f60deb8...e700`, v3 direct attempt_002 tree `a5ede6b1...830d`, and v3 factorized attempt_002 tree `28f159a5...0da1`. Full paths, hashes, counts, and byte sizes are in the JSON evidence.
- GPU inventory query, CUDA, model fit/sample, data generation, external validation/test, TSTR, privacy, and full run counts: all zero.
