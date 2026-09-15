# CoF-HCMTTPP-v2 H1 attempt_003 read-only forensic

Date: 2026-08-06 (Asia/Seoul)
Source commit recorded by both attempts: `8a753e6eed0dbd49b30f93968d98712472a90c4c`
Relevant execution-source SHA-256: `f5b7ccb42d8e1e430ef2da7945c9f13c0b4ae013ee20804ff97c59c0693581b9`
Runner-config SHA-256: `fa0ab8cf2e6fd70cfd7dbb43a530f47bbeab5fa59a0ae5aae3923b3edf054329`
Model SHA-256: `f8b73074934b4bb9e2ea46de5cfdd3e35b447b23c9a1f8302b87c4182c0dcbf6`

## Scope and non-execution statement

This analysis read only the two stored attempt_003 trees, their worker terminals, launch logs, checkpoint bundles, and the source that defines their serialization and failure guard. It did not query a GPU or CUDA, read an external-data body, import a model for execution, train, regenerate, sample, evaluate, access internal test or Sparkov fraudTest, or create an attempt_004 authorization or launch plan. No existing runtime artifact was changed.

The checkpoint inspection used CPU-only deserialization of the already stored checkpoint dictionaries. It did not reconstruct a training batch or call model forward/backward/evaluation.

## Executive finding

AMLSim reached a valid, fully finite checkpoint and progress event at update 14,000, then raised `InvalidH1GapStateError: H1 gap parameters are non-finite or invalid` before the update-14,100 progress event. The exact detection step is therefore in `[14,001, 14,099]`. The exact update that first created a non-finite value can be earlier than the detection step within that interval because the training loop checks the scalar loss before backward but does not persist or check gradients and parameters after every optimizer step.

The failure site proves that the composite `H1GapParameters.invalid_mask` became true before the gap NLL for the failing batch was produced. It does **not** identify which member of that composite was first. The failed batch's backbone hidden state, hurdle logit, RQS height/derivative outputs, tail gate, `r_beta`, beta, gradients, and post-step parameters were not persisted. Component-level attribution beyond this boundary is therefore not recoverable from the authorized stored artifacts.

The exact state classification is:

> **b) H1 numerical-stability/architecture limitation**

The frozen family state is:

> **STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL**

This classification is supported because the failure arose in the model's training-time H1 gap-parameter validity contract after 14,000 updates, while the artifact/runner chain remained intact and the same source completed 20,000 Sparkov updates. There is insufficient evidence to call a specific formula or module an implementation bug. The narrower root component remains **INCONCLUSIVE**.

## Artifact integrity

| Dataset | Terminal | Attempt tree SHA-256 | Files | Bytes | Terminal SHA-256 | Index SHA-256 | Checksum SHA-256 | Full chain |
|---|---:|---|---:|---:|---|---|---|---|
| AMLSim | FAILED | `fa1648c27cba02e6601677c61645627012fac1df505f5e726f7c0de28f93799e` | 27 | 233,903,239 | `81e450838eaa59c072edbc766804372d45c72af4fbd381f61766f3a9b095723a` | `855c3ec5fdb27b5868f1e53d5eef5b5018eb169ef003d631c3743431e1c5dad3` | `4b1f308798641885bb9c89ebf2af548bf82d5012cf4ff8f1026f7e905157097b` | PASS |
| Sparkov | COMPLETE | `41b9fb4b2de9c2a5780127bd29ea0f38e07714bfa1859eb1031a8001708f5681` | 41 | 63,013,817 | `01e6b08dcfb3b248622046c7b07681b6af8f15409b2c0ca2604d75640fa56e67` | `568058a8027e3e2edb408e015b05a922dfb172851f9e5a17ae28c783967c6fd3` | `b778208ce364105b1b53051c555683a074fd10a9454dfb32bee1a84280e6b18d` | PASS |

For both datasets:

- the terminal's index and checksum references equal the actual file hashes;
- every indexed file exists and matches its checksum entry;
- the checksum manifest contains exactly the indexed files plus `artifact_index.json`;
- no unexpected file is present in the attempt tree;
- the dataset worker terminal points to the exact candidate terminal hash and has the same status.

Worker-terminal SHA-256 values are `0ccf4624109bb22f711938e6f9d7b8fc211109c7d9f09b22fee02b1ef9eb05d1` (AMLSim FAILED) and `766f912108fb4c612b5cdc0b89008bbb6108d5cbcd09e364546e6af566d8de0c` (Sparkov COMPLETE).

## AMLSim failure boundary

| Evidence | Last finite / first invalid observation |
|---|---|
| Progress | update 14,000, elapsed 274.776499301 s, total loss `-0.4651280343532562`, finite |
| Checkpoint | `checkpoints/step_014000.pt`, SHA-256 `fc45f0ba44ebaade455018889c5fa3f1972aee1105641b233ef6ecaa66a88b8b` |
| Checkpoint state | 56 state entries, 4,169,827 elements, all floating tensors finite; state-dict SHA-256 `1116bc4b776b0fba3ac620c0c3d1952db8590c9859c9113cf86a3546a7dc9623` |
| First invalid observation | after update 14,000 and before the missing update-14,100 progress event; exact detection step in `[14,001, 14,099]` |
| Terminal | elapsed 277.671623438 s, `child_exception`, `InvalidH1GapStateError` |
| Diagnostics | no last finite component diagnostics persisted; final `diagnostics.json` is only `{"status":"FAILED"}` |
| Gradients/optimizer | not present in checkpoint schema and not present in progress; finite status at the failing boundary is unknowable |

The checkpoint bundle contains only `model_config`, `provenance`, `provenance_sha256`, `schema_version`, `state_dict`, `state_dict_sha256`, and `tail_state`. Consequently, the failed batch cannot be replayed exactly from the artifact without forbidden data-body access and execution, and its gradients/optimizer state cannot be recovered at all.

### Component localization

`H1GapDecoder._require_valid` combines these conditions into one mask: non-finite hurdle logits; non-finite RQS height fractions; non-finite RQS derivatives; non-finite or boundary-saturated `pi_tail`; and non-finite or non-positive beta. The RQS widths are fixed uniform quantile widths, not a learned per-event tensor.

| Candidate tensor/module | Last persisted evidence | Failure-time evidence | Finding |
|---|---|---|---|
| Backbone hidden state | all backbone state tensors finite at 14,000; parameter abs-max 4.94987 | hidden activations not stored | INCONCLUSIVE; evidence against a pre-existing checkpoint non-finite |
| Hurdle logit | head weights/bias finite; abs-max 1.05280 | logits not stored | INCONCLUSIVE |
| RQS widths | fixed uniform by implementation | not dynamic | not the first dynamic tensor |
| RQS heights/derivatives | spline-head state finite; abs-max 4.64241 | transformed outputs not stored | INCONCLUSIVE |
| Tail gate | residual-head state finite; abs-max 0.625688 | `r_tail`/`pi_tail` not stored; exact 0/1 saturation is part of the composite guard | INCONCLUSIVE |
| `beta_base` | finite and positive, 0.5465437063680696 for both Y classes | immutable train-only buffer | REFUTED as the non-finite source |
| `r_beta` / `exp(r_beta)` / beta | tail-scale head finite; abs-max 0.00240103 | `r_beta` and beta not stored | INCONCLUSIVE |
| Loss terms | total loss finite through update 14,000 | failing `event_nll` aborted before a scalar loss was assembled | no first non-finite loss term was observed |
| Gradients/parameters | checkpoint parameters finite at 14,000 | gradients and post-14,000 parameter states not persisted | INCONCLUSIVE |

## Same-step Sparkov comparison

At update 14,000 Sparkov had finite total loss `16.822757720996` and a fully finite checkpoint. It continued to update 20,000, where its final checkpoint also contained 56 finite state entries and had state-dict SHA-256 `89168a8f7a0030f7f0ed0d77a90aa60e2e1c1912c4f7242c97347fd0f96d4fc4`.

| Module state at update 14,000 | AMLSim abs-max | Sparkov abs-max | Both finite |
|---|---:|---:|---|
| backbone | 4.94987 | 5.34985 | yes |
| hurdle head | 1.05280 | 1.01865 | yes |
| RQS head | 4.64241 | 9.07358 | yes |
| tail-gate residual head | 0.625688 | 2.68618 | yes |
| tail-scale residual head | 0.00240103 | 0.00626374 | yes |
| amount path | 5.20660 | 4.63231 | yes |
| receiver head | 2.42885 | 1.79221 | yes |

Sparkov's RQS, tail-gate, and tail-scale parameter maxima were all larger than AMLSim's at the matched checkpoint, yet Sparkov completed. This is evidence against raw head-weight magnitude alone being a sufficient explanation. It does not rule out dataset-conditioned activations or a transient gradient event in AMLSim.

## Hypothesis decisions

| Hypothesis | Decision | Evidence |
|---|---|---|
| Unbounded `exp(r_beta)` or tail likelihood caused overflow/gradient explosion | INCONCLUSIVE | AMLSim's persisted beta base and tail-scale head are finite and small, but failed-batch `r_beta`, beta, tail NLL, and gradients are absent. Sparkov completed with a larger tail-scale head and final mean beta/baseline ratios near 1.001. |
| RQS density/Jacobian gradient instability | INCONCLUSIVE | AMLSim's RQS state is finite through 14,000 and no RQS gradient is stored. The decreasing negative total loss is compatible with increasing continuous-density concentration but is not component-specific proof. Sparkov completed with larger RQS weights. |
| Amount/receiver/backbone non-finite propagated into gap head | INCONCLUSIVE, weak evidence against | Every corresponding checkpoint tensor is finite at 14,000, but the failed hidden activation and gradients were not stored. AMLSim's much larger receiver parameterization can alter shared-backbone gradients, but this artifact set cannot establish that path. |
| Evaluator/artifact/runner defect | REFUTED | The exception occurred during training `event_nll`, before sampling/evaluation. Both integrity chains pass, the failure is terminalized correctly, and Sparkov completed under the same source. |

No hypothesis about the first dynamic tensor is upgraded to SUPPORTED because doing so would require telemetry that was not persisted or a forbidden rerun.

## Sparkov COMPLETE verification

- `evaluation.json`: `VALID`; mask, padding, and train discrete support all `true`.
- Forbidden access: `internal_test_accessed=false`, `sparkov_fraud_test_accessed=false`; diagnostic access counts are both zero.
- Runtime: requested/actual updates 20,000/20,000; 340.612837065 s; peak memory 1,041,697,280 bytes.
- `gate_decision.json`: `FAIL`, family state `STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`. Completion means execution and hard validity completed; it does not mean the scientific gate passed.
- Gate values: gap KS Y0/Y1 `0.1994332381759053` / `0.11033099297893678`; positive-gap KS Y0/Y1 identical; receiver TV Y0/Y1 `0.055170394537336215` / `0.24849548645937813`; full receiver TV `0.054898847631242`; coherence error Y0/Y1 `0.00047499271317996824` / `0.0005177323323841573`.
- Density diagnostics: 195,250 finite NLL/density events, zero non-finite samples, zero hard upper clips, positive-CDF total-mass error 0.0.
- Terminal/index/checksum and worker-terminal chains all PASS as documented above.

## Final boundary

This forensic does not authorize H1 attempt_004, any H1 retry, internal-test access, or Sparkov fraudTest access, and it does not establish a repair target. AMLSim has a localized training-time numerical failure in the H1 gap-parameter boundary, but the stored schema is insufficient to identify the first offending tensor or gradient. Any future decision would require a separately preregistered diagnostic-instrumentation scope; this report creates none. The final state remains `STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`.
