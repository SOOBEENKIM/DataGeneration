# CS-SAF 세 목적함수 비교 결과 — 2026-09-16

**등록된 진단은 완료됐지만 U/A/B 모두 5% 조건의 비활성 반응 기준은 FAIL이다.**
보조 손실의 전체 비중과 집단 배분을 분리해 비교했다. 의존성이 없는 희소
집단에서 전역 평균 보조 손실 추가와 집단 균형화가 각각 반응을 키웠다.
일반 손실의 실패도 학습·검증 집합에 넓게 나타났다. 손실 변경만으로 문제를
해결했다고 볼 수 없으며, 후속 비율·여러 시드 실험으로 확대하지 않았다.

[사전등록](loss_control_v1_preregistration.md) ·
[전체 수치·검증 기록](loss_control_v1_result.json) · [최신 상태](STATUS.md).

## 통제한 비교와 실행 검증

| 이름 | 목적함수 | 비교에서의 역할 |
|---|---|---|
| U / CS2-U1 | 기존 base NLL | 일반 목적함수 |
| A / CS2-A1 | base + 전체 train transition 평균 repeat BCE | 보조 손실 추가 효과 |
| B / CS2-B1 | base + 집단별 transition 평균 repeat BCE의 균등 평균 | 집단 배분 효과 |

V2의 집단별 rank-16 경로 두 개, 총 133,549개 파라미터를 유지했다.
Seed 20260930, 초기 tensor, epoch별 entity 순서, 데이터·tensorizer, AdamW,
배치 크기, 학습 예산, global validation base NLL checkpoint 선택을 고정했다.
Auxiliary는 train에서 고정한 분모를 사용하며 minibatch 집단 비율로
재정규화하지 않는다. A/B 모두 auxiliary 계수는 1이다.

Train N=31,951, E=766,935 events, T=734,984 transitions이며
희소 집단 T_1=35,542(4.83575%)다. 전체 데이터 기준 direct-route의
집단별 손실 계수는 다음과 같다.

| 목적함수 | 다수 집단 | 희소 집단 | 합 |
|---|---:|---:|---:|
| U | 0.911996 | 0.046343 | 0.958339 |
| A | 1.863639 | 0.094700 | 1.958339 |
| B | 1.411996 | 0.546343 | 1.958339 |

A/B의 합은 같고 배분이 다르다. 이는 gradient norm이나 AdamW update를
같게 만든다는 뜻이 아니다. Base의 기존 minibatch event 평균과 shared
feature 업데이트는 유지했으므로 그 영향을 제거한 인과 분해도 아니다.

- 사전등록 commit: `d4069c9`; 구현·CPU·GPU source:
  `f4bed4b9a17970d2cd146515f3e97d1ae0db6947`.
- **관련 테스트 82개 PASS**. A의 값·gradient·고정 분모, U/B의 기존
  forward·loss·gradient 동일성, entity 단위 진단 집계를 검사했다.
- **CPU 6회, 모든 gate PASS**. U/A/B 각각 두 번의 이력·최적 가중치가
  동일했고 checkpoint reload, 생성·zero-gap 검사도 통과했다.
  Train objective 감소는 각각 48.3120%, 48.3447%, 48.3266%다.
  Tiny CPU set의 최적 epoch는 0이며 일반화 성공을 의미하지 않는다.
- **GPU 6 fits 완료**: pi=.05, kappa=0/1 × U/A/B. 기존 trained weight를
  가져오지 않고 새로 학습했다. RTX 3090 GPU 0/1/2를 사용했다.
- 기존 U/B 네 fit은 initial 및 best tensor가 **bitwise 동일**했고,
  선택 epoch·NLL·full-support 반응도 이전 v2 결과와 정확히 같았다.
- Train 31,951개(희소 1,545개), validation 6,846개(희소 331개)를 사용했다.
  Held-out test와 realized oracle latent는 이 실행에서 읽지 않았다.

## 모든 조건의 반응 결과

모든 validation nonfirst history에서 31개 train support bin에 대한
max-minus-min을 계산하고, history→entity→집단 순으로 평균했다.
아래는 **copy 확률** 반응이며 비활성 상한은 .05다.

| 조건 | U | A | B |
|---|---:|---:|---:|
| kappa=0, 다수 비활성 | 0.020427 | 0.024214 | 0.019987 |
| kappa=0, 희소 비활성 | **0.055189** | **0.061733** | **0.081949** |
| kappa=1, 다수 비활성 | 0.025898 | 0.024694 | 0.033930 |
| kappa=1, 희소 활성 | 0.365049 | 0.344217 | 0.343783 |

관측 가능한 **repeat 확률**도 같은 실패다.

| 조건 | U | A | B |
|---|---:|---:|---:|
| kappa=0, 다수 비활성 | 0.020103 | 0.023822 | 0.019661 |
| kappa=0, 희소 비활성 | **0.054313** | **0.060732** | **0.080610** |
| kappa=1, 다수 비활성 | 0.025490 | 0.024293 | 0.033385 |
| kappa=1, 희소 활성 | 0.359321 | 0.338652 | 0.338277 |

세 목적함수 모두 active >=.05, active >=2*max null, zero-gap 불변성,
생성 유효성은 통과했다. 실패는 **의존성이 없는 희소 집단의 copy/repeat
반응이 .05를 초과한 것**이다. 한 시드·한 비율의 국소 사전 점검이며
전체 비율 파일럿 통과로 해석할 수 없다.

Kappa=0 희소 validation의 paired copy 차이는 A-U **+0.006544**,
B-A **+0.020216**이다. 기술적 entity 표준오차는 각각 .000151/.000332다.
이는 고정한 학습 모델 아래 entity 변동이며 여러 학습 시드의 불확실성이
아니다. 성능 우위의 유의성 검정이나 일반적인 인과 효과를 주장하지 않는다.

동일한 0-based epoch 9에서도 U/A/B 반응은 **.054966/.060867/.081949**로
같은 순서다. 선택 checkpoint의 시점 차이만으로 설명되지는 않는다.

## 일반 손실 U의 실패에 대해 확인된 것

Kappa=0 희소 집단의 U 결과는 다음과 같다.

| 진단 | Train | Validation |
|---|---:|---:|
| 전체 gap 구간 copy 반응 | 0.054936 | 0.055189 |
| Train에서 정한 중앙 5–95% bin 구간 반응 | 0.053865 | 0.054159 |
| Entity 평균 반응이 .05를 넘는 비율 | 89.06% | 90.33% |

따라서 이 실패를 validation 일부 이상치나 극단 gap bin만의 현상으로
설명하기 어렵다. 중앙 구간은 train 집단별 gap 빈도로만 정했고 validation에
그대로 적용했다. 기존 전체 구간 기준을 중앙 구간으로 대체하지 않았다.

중요한 제약도 확인됐다. 같은 U 가중치에서 현재 gap 경로를 통째로 끄면
희소 validation의 repeat BCE가 **.489599 → .542023**으로 악화된다.
Train에서도 .485774 → .540038로 악화된다. A/B에서도 경로를 끄면 이
손실이 악화된다. 따라서 "이 경로는 쓸모없는 잡음이므로 없애면 된다"는
해석은 이번 고정 checkpoint 진단과 맞지 않는다.

다만 zero-gap ablation은 재학습 없이 경로 전체를 제거한다. 경로 안에
담긴 gap과 무관한 history 보정까지 제거할 수 있다. 이 결과가 null
조건에 진짜 gap 신호가 있다는 증거나 모델의 불필요한 반응을 정당화하는
증거는 아니다. 학습 과정의 유일한 원인도 아직 식별하지 못했다.

## 실제 반복 예측 손실은 반응 크기와 구분해야 한다

아래는 best checkpoint의 집단별 **validation factual repeat BCE**다.
단위는 nats, entity 내부 transition 평균 후 entity 평균이며 작을수록 좋다.

| 조건 | U | A | B |
|---|---:|---:|---:|
| kappa=0, 다수 비활성 | 0.492823 | 0.491865 | 0.492588 |
| kappa=0, 희소 비활성 | 0.489599 | 0.488560 | 0.495011 |
| kappa=1, 다수 비활성 | 0.495195 | 0.494403 | 0.495405 |
| kappa=1, 희소 활성 | 0.451531 | 0.453562 | 0.447482 |

A는 희소 null 반응을 키우면서도 그 셀의 factual BCE는 U보다 조금 낮다.
B는 A보다 희소 null BCE가 **.006451 높지만**, 활성 집단에서는
**.006080 낮다**. 그러므로 균형화가 모든 측면에서 나쁘다고 말할 수는 없다.
이 역시 선택된 checkpoint/한 시드의 탐색적 비교이며, conditional TV나
oracle에 대한 분포 오차 전체를 평가한 결과가 아니다.

Kappa=0 희소 train BCE는 U/A/B .485774/.482821/.466983이다.
B는 train에서 더 낮지만 validation에서는 더 높아, 희소 집단 fitting과
일반화 사이의 문제와 양립한다. 그것만으로 U의 넓은 null 반응까지
설명되지는 않는다. 고정 epoch 9에서도 B-A의 해당 validation BCE 차이는
+.005283이다. 활성 조건에서 A-U의 BCE 차이는 best와 epoch 9의 부호가
달라, 모든 손실 비교를 checkpoint 선택과 독립적인 효과로 볼 수 없다.

| 학습 | 실행 epochs | 선택 epoch (0-based) | Global validation base NLL |
|---|---:|---:|---:|
| kappa=0 U | 16 | 10 | 8.219602 |
| kappa=0 A | 16 | 10 | 8.219443 |
| kappa=0 B | 15 | 9 | 8.221156 |
| kappa=1 U | 16 | 10 | 8.194628 |
| kappa=1 A | 14 | 8 | 8.193920 |
| kappa=1 B | 16 | 10 | 8.195608 |

## 다음 연구의 출발점과 경계

이번 진단은 "balanced 보조 손실만 제거하면 해결된다"거나 "전체 비중만
통제하면 해결된다"는 방향을 지지하지 않는다. 연구 확대를 위한 모델
성공은 아직 아니다. 모든 실패를 보존하고 새로운 가설을 구체화해야 한다.

다음의 좁은 가설은 **현재 bilinear 경로가 history 보정과 gap에 따른 변화를
함께 담당할 가능성**이다. 식 자체는 임의의 고정 train gap 분포에서

```text
r(c,g) = mean_g_train[r(c,g)] + (r(c,g) - mean_g_train[r(c,g)])
```

로 나눌 수 있다. 첫 항은 history에 의존하지만 현재 gap에는 의존하지
않는다. Zero-gap 전체 제거는 두 항을 함께 없앤다. **이 분해의 각 항이
실제 실패에 기여하는 정도는 이번 등록에서 측정하지 않았다.**

후속 단계로는 먼저 저장된 checkpoint에서 train 기준 평균 성분을
유지한 채 gap에 따른 잔차만 제거하는 진단을 별도로 등록할 수 있다.
그 결과가 뒷받침하면 history 보정과 중심화한 gap 잔차를 분리하고 잔차만
수축하는 모델을 별도 후보로 설계할 수 있다. 단순한 수학적 중심화만으로
표현력이 제한되거나 null 반응이 자동 해결되는 것은 아니다.
이 보고서 작성 시 후속 진단·모델은 제안 단계였다. 이후 진단은 별도로 사전등록하고
[실행을 완료했다](route_decomposition_v1_report_2026_09_16.md): 평균 성분은 유용했고,
null residual 제거는 validation을 개선하지만 train을 악화시켰다. 수정 모델은 여전히
**제안이며 사전등록·구현·학습 전**이다.

수정 후보가 .05 및 등록된 여러 비율의 반응 기준을 통과해야 그다음으로
갈 근거가 생긴다. 이후에도 정확한 조건부 분포 오차 집계와 train-only
noninferiority margin을 먼저 고정하고, 여러 시드에서 정확도·안정성을
검증해야 한다. 이번 진단 완료만으로 대규모 실험을 시작하지 않았다.

## 보존과 재현

총 **12,288개 entity / 283,986개 nonfirst gap**을 생성했다. Support 위반과
예약 mark 출력은 0, numeric value는 모두 유한하며 zero-gap 반응은 정확히
0이다. Support는 empirical train atoms이며 population support 전체가 아니다.

**62개 artifact checksum 비교**, checkpoint/sample 모델 identity,
entity array와 집계 수치의 일치를 재검증했다. 각 fit의 best·epoch 9
checkpoint, epoch 이력, 생성 sample, train/validation entity 진단 배열을
원격에 보관한다. 코드·불변 계약·이 문서·작은 결과 JSON은 Git에 보존한다.

- 계약 SHA: `1de81bc0f0af1fe0ac829a1f24fe82a58c88348212e66f74b25e230a1ead0cdd`.
- Python: `/home/finx_sbk/.conda/envs/cofseq/bin/python3`, PyTorch `2.1.2+cu121`.
- CPU: `artifacts/cs_saf/loss_control_v1/cpu_v1/`.
- GPU: `artifacts/cs_saf/loss_control_v1/gpu_v1/`.

기록된 source의 clean checkout에서 새 출력 경로를 사용한다:

```bash
python -m scripts.run_cs_saf_loss_control cpu \
  --cache-root artifacts/cs_saf/prepared_v1 \
  --output artifacts/cs_saf/loss_control_v1/cpu_reproduction
python -m scripts.run_cs_saf_loss_control run \
  --cache-root artifacts/cs_saf/prepared_v1 \
  --cpu-gate artifacts/cs_saf/loss_control_v1/cpu_reproduction/COMPLETE.json \
  --output artifacts/cs_saf/loss_control_v1/gpu_reproduction --gpus 0,1,2
```

추가 재현을 수행한 것은 아니다. YAML의 등록 당시 상태는 역사적 metadata로
고정돼 있으며 현재 실행 상태는 이 보고서와 STATUS를 따른다.
