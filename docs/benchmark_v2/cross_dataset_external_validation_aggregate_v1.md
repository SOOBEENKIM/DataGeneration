# AMLSim and Sparkov cross-dataset validation aggregate

This is a read-only join of the frozen AMLSim and Sparkov validation aggregate
artifacts. It does not read runtime evaluations, samples, checkpoints, frozen
arrays, internal test, or Sparkov `fraudTest`, and it performs no model or GPU
operation.

Inputs:

- AMLSim aggregate JSON SHA-256:
  `5363efdbac2ac2f2a72bc4fdacac23c021cb27af34f106cf24aff1143b1e1197`
- Sparkov aggregate JSON SHA-256:
  `ed6b77280f1f62a9cff3f9d77116effd5f1bc7e4956e2f48b321d795b6c4f524`

All scores are lower-is-better. “Macro” is the unweighted arithmetic mean of
the two stored dataset values. It is descriptive only and is not a new model
selection, threshold, or test-unlock rule.

| Model | AML fidelity | Sparkov fidelity | Macro fidelity | AML coherence | Sparkov coherence | Macro coherence | AML combined | Sparkov combined | Macro combined | Macro rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| empirical IID | 1.8783 | 2.0965 | **1.9874** | 13.7394 | 5.0194 | 9.3794 | 7.8088 | 3.5579 | **5.6834** | 1 |
| CTGAN | 12.3592 | 10.4863 | 11.4227 | 13.7514 | 3.9409 | 8.8462 | 13.0553 | 7.2136 | 10.1344 | 2 |
| frozen non-v3 CoF | 12.3967 | 13.0634 | 12.7300 | **12.9044** | **3.8579** | **8.3812** | 12.6506 | 8.4606 | 10.5556 | 3 |
| TVAE | 6.8964 | 48.6941 | 27.7953 | 13.1489 | 429.9706 | 221.5598 | 10.0226 | 239.3324 | 124.6775 | 4 |

The stored validation evidence is heterogeneous across datasets. IID has the
lowest macro fidelity and combined score, while CoF has the lowest macro
coherence. This table supports descriptive comparison only; it does not alter
either dataset’s original validation conclusion.

Machine-readable copies are
`cross_dataset_external_validation_aggregate_v1.json` and `.csv`.
