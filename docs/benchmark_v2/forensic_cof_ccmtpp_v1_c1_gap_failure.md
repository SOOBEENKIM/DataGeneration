# CCMTPP v1 C1 Y=0 gap-KS failure forensic

- Source HEAD (read-only): `b3f196a958850110cdbd4dddd3d325398ddc0c3c`
- Final chain state: `STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL`
- 제한된 최종 결론: **positive-gap conditional calibration 문제**
- 분석 입력은 저장된 C0/C1 validation sample, frozen train/validation, train-only transform/gap edges/tau와 저장 metric evidence뿐이다. 모델·evaluator runner는 import하거나 재실행하지 않았다.

## 결론

두 데이터셋에서 공통으로 재현되는 원인은 positive-gap 조건부분포의 calibration 실패다. AMLSim은 zero atom 불일치도 결정적이지만 Sparkov에는 zero atom이 전혀 없는데도 Y=0이 실패한다. C1 positive-only Y=0 KS는 AMLSim 0.773860, Sparkov 0.069522로 각각 C0의 0.547556, 0.042308보다 악화됐다. 따라서 AMLSim 전용 zero hurdle만으로 공통 원인을 설명할 수 없다.

train→validation shift와 evaluator bin mapping은 주원인에서 배제된다. Y=0 train→validation KS는 AMLSim 0.0276605, Sparkov 0.0127674로 C1 오류의 각각 11.7%, 18.4%에 불과하다. continuous C1을 frozen bin/tau로 사상하면 Y=0 KS가 각각 0.0212473, 0.0180851 감소하므로 mapping은 실패를 만들지 않고 일부를 가린다.

## Classwise gap 분해

| Dataset | Y | train→val KS | C0 KS | C1 KS | val zero | C0 zero | C1 zero | positive-only C0 KS | positive-only C1 KS | C1 KS supremum | signed CDF Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amlsim | 0 | 0.0276604573 | 0.227623598 | 0.23570909 | 0.653988058 | 0.881611656 | 0.418278968 | 0.547556451 | 0.773860396 | 0 | -0.23570909 |
| amlsim | 1 | 0.0260528591 | 0.466188368 | 0.203175088 | 0.523278889 | 0.144100137 | 0.340711342 | 0.277135055 | 0.585089141 | 0.99999845 | 0.203175088 |
| sparkov | 0 | 0.0127674409 | 0.0423084565 | 0.0695224352 | 0 | 0 | 0 | 0.0423084565 | 0.0695224352 | 81520.000 | -0.0695224352 |
| sparkov | 1 | 0.0111846188 | 0.300651956 | 0.15220662 | 0 | 0 | 0 | 0.300651956 | 0.15220662 | 9405.000 | -0.15220662 |

CDF Δ는 `F_C1 − F_validation`이다. AMLSim Y=0 supremum은 gap=0에서 발생하며 zero-atom 부족량과 정확히 같다. Sparkov Y=0 supremum은 gap=81,520에서 발생하고 C1 누적질량이 0.0695224 낮다.

## Positive-gap 분위수 차이

아래는 C1−validation 차이다. JSON/CSV에는 q10/q25/q50/q75/q90/q95/q99와 C0 비교가 모두 들어 있다.

| Dataset | Y | q10 Δ | q25 Δ | q50 Δ | q75 Δ | q90 Δ | q95 Δ | q99 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| amlsim | 0 | -0.999717628 | -0.999275397 | -0.998267497 | -1.00280541 | -0.133666801 | -7.98866501 | 0 |
| amlsim | 1 | -0.99961187 | -0.998891285 | -1.99546132 | -0.0478074551 | -0.0531929016 | 0 | 0 |
| sparkov | 0 | -981.992224 | 1129.334 | 1316.967 | 2537.311 | 35530.973 | -0.015625 | -0.015625 |
| sparkov | 1 | 1989.687 | 4868.398 | 7037.240 | 3785.181 | -32.7171875 | -0.015625 | -0.015625 |

AMLSim C1은 양의 gap임에도 q50이 Y=0에서 0.00173, Y=1에서 0.00454로 validation의 1 및 2보다 매우 작다. Sparkov Y=0은 q90이 validation보다 35,530.97 커서 상위 꼬리를 과대 생성한다.

## Frozen edge/bin mapping과 support boundary

| Dataset | Y | raw KS | mapped-tau KS | mapped−raw | changed fraction | quantization MAE | lower-bound rate | upper-bound rate | first bin | last bin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amlsim | 0 | 0.23570909 | 0.214461778 | -0.0212473114 | 0.575577105 | 0.125666935 | 0.418278968 | 0.00613869366 | 0.868449836 | 0.0122145874 |
| amlsim | 1 | 0.203175088 | 0.203175088 | 0 | 0.624332163 | 0.194232187 | 0.340711342 | 0.0349564952 | 0.726453976 | 0.0735765532 |
| sparkov | 0 | 0.0695224352 | 0.0514372954 | -0.0180851398 | 1 | 1685.949 | 0 | 0.0549978563 | 0.10527444 | 0.108787945 |
| sparkov | 1 | 0.15220662 | 0.126880642 | -0.0253259779 | 1 | 1975.969 | 0 | 0.092778335 | 0.0855065196 | 0.183049147 |

전체 train-edge/bin별 train·validation·C0·C1 질량과 각 validation 차이는 CSV의 `train_edge_bin_mass` 및 JSON의 `train_edge_bin_mass`에 고정했다. 재계산한 C1 bin은 저장 `dt_bin`과 두 데이터셋 모두 완전히 일치한다.

## 보존된 C1 개선

| Dataset | Scope | Metric | C0 | C1 | absolute Δ | relative Δ | 판정 |
|---|---|---|---:|---:|---:|---:|---|
| amlsim | y0 | gap_ks | 0.227623598 | 0.23570909 | 0.00808549164 | +3.55% | worsened |
| amlsim | y0 | short_gap_receiver_repeat_error | 0.213971814 | 0.096367415 | -0.117604399 | -54.96% | improved |
| amlsim | y0 | receiver_tv | 0.739997802 | 0.567449747 | -0.172548055 | -23.32% | improved |
| amlsim | y1 | gap_ks | 0.466188368 | 0.203175088 | -0.26301328 | -56.42% | improved |
| amlsim | y1 | short_gap_receiver_repeat_error | 0.108702532 | 0.0462025316 | -0.0625 | -57.50% | improved |
| amlsim | y1 | receiver_tv | 0.933597924 | 0.812395054 | -0.12120287 | -12.98% | improved |
| amlsim | all | full_receiver_tv | 0.734645861 | 0.562529727 | -0.172116134 | -23.43% | improved |
| amlsim | all | head_receiver_tv | 0.624824388 | 0.409749161 | -0.215075227 | -34.42% | improved |
| amlsim | all | tail_receiver_tv | 0.768493626 | 0.60257265 | -0.165920976 | -21.59% | improved |
| amlsim | all | unk_rate_error | 0.018508961 | 0.018508961 | 0 | +0.00% | unchanged |
| sparkov | y0 | gap_ks | 0.0423084565 | 0.0695224352 | 0.0272139787 | +64.32% | worsened |
| sparkov | y0 | short_gap_receiver_repeat_error | 0.000501980936 | 0.000296870446 | -0.00020511049 | -40.86% | improved |
| sparkov | y0 | receiver_tv | 0.0679173071 | 0.062882329 | -0.0050349782 | -7.41% | improved |
| sparkov | y1 | gap_ks | 0.300651956 | 0.15220662 | -0.148445336 | -49.37% | improved |
| sparkov | y1 | short_gap_receiver_repeat_error | 0.000517732332 | 0.000258866166 | -0.000258866166 | -50.00% | improved |
| sparkov | y1 | receiver_tv | 0.327983952 | 0.242978937 | -0.085005015 | -25.92% | improved |
| sparkov | all | full_receiver_tv | 0.0655057618 | 0.0624379001 | -0.00306786172 | -4.68% | improved |
| sparkov | all | head_receiver_tv | 0.0616622268 | 0.0602358421 | -0.00142638461 | -2.31% | improved |
| sparkov | all | tail_receiver_tv | 0.0822069962 | 0.0729127058 | -0.00929429046 | -11.31% | improved |
| sparkov | all | unk_rate_error | 0 | 0 | 0 | NA | unchanged |

C1은 두 데이터셋에서 Y=1 gap, Y=0/Y=1 coherence, classwise/full receiver TV를 개선했다. 유일한 sequential gate blocker는 두 데이터셋의 Y=0 gap KS 비열화다.

## 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| zero-inflated hurdle gap head 필요 | dataset-specific support only | AMLSim zero atom에는 필요성이 보이지만 zero atom이 없는 Sparkov 실패를 설명하지 못한다. |
| positive-gap conditional calibration 문제 | **SUPPORTED / 최종 선택** | positive-only Y=0 KS가 두 데이터셋 모두 C0보다 악화됐다. |
| train/validation shift가 주원인 | REFUTED | Y=0 shift는 C1 오류보다 AMLSim 8.52배, Sparkov 5.45배 작다. |
| evaluator bin mapping이 주원인 | REFUTED | mapping 후 Y=0 KS가 감소해 raw continuous 오류 일부를 가린다. |
| INCONCLUSIVE | 배제 | 두 데이터셋에 공통인 positive-gap 증거가 충분하다. |

## Integrity와 금지 상태

- amlsim: C0 index 211 entries PASS; C1 checksum 56 entries 및 index 55 entries membership PASS.
- amlsim: C0/C1 mask·padding·train discrete support PASS; C1 test_accessed=false, sparkov_fraud_test_accessed=false.
- sparkov: C0 index 211 entries PASS; C1 checksum 56 entries 및 index 55 entries membership PASS.
- sparkov: C0/C1 mask·padding·train discrete support PASS; C1 test_accessed=false, sparkov_fraud_test_accessed=false.
- C2/C3/C4 authorization·실행과 새 candidate tuning은 금지 상태다.
- GPU/CUDA, 모델 import/fit/sample, evaluator runner 재실행, internal test, Sparkov fraudTest, 새 architecture 구현은 수행하지 않았다.

## 최종 상태

`STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL`
