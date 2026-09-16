# CS-SAF v2 구현 및 등록 파일럿 결과 — 2026-09-16

**최종 판정: v2 파일럿 FAIL, pi=0.05에서 종료.** 집단별 경로 분리는
의존성 있는 데이터의 비활성 반응을 줄였지만, 의존성이 없는 데이터의
희소 집단에서 불필요한 반응을 키웠다. 구조 변경만으로는 전체 기준을
만족시키지 못했다. 개선된 한 셀을 전체 성공으로 해석하지 않는다.

등록된 순서대로 구현 → 구조 검사 → CPU gate → 원격 GPU 파일럿을
완료했다. 사전등록 이후 loss, 초기화, rank, seed, 기준을 변경하지 않았다.
기존 v1의 실패 결과와 checkpoint도 그대로 보존했다.

전체 수치와 provenance: [v2_pilot_v1_result.json](v2_pilot_v1_result.json).
실행 전 등록: [v2 preregistration](revision_v2_preregistration.md).

## 구현과 검증

`models/cs_saf_v2.py`의 CS2-U1/CS2-B1은 공유 rank-32 bilinear 경로를
관측 static code로 선택하는 rank-16 경로 두 개로 바꾼다. 어떤 label이
활성인지 알려주는 mask나 kappa 입력, oracle target은 없다.
GRU, static/gap embedding, base copy logit, fresh-mark 및 value head는
공유한다. 기존 목적함수와 데이터 표현은 유지했다.

- 경로 파라미터 **5,440개**, 전체 **133,549개**로 v1과 동일하다.
- Seed 20260930으로 새로 초기화한 v1과 공통 tensor가 bitwise 동일하다.
  Trained v1 checkpoint를 초기값으로 사용하지 않았다.
- 새 경로는 별도 CPU generator seed 20261010과 등록된 draw order로
  초기화했다. 두 집단의 interaction weight는 모두 0에서 시작한다.
- CS2-U1/B1의 초기 state와 minibatch 순서가 같고, 차이는 기존과 같은
  balanced observable-repeat 보조 손실의 유무다.
- **관련 테스트 71개 PASS**. 다른 집단 bank로의 직접 gradient가 정확히
  0인지, 각 bank가 자신의 gradient를 받는지, label/embedding/bank 동시
  교환 시 예측이 유지되는지, strict past, likelihood와 audit 일치,
  support, checkpoint reload와 고정 loss 분모 등을 확인했다.

공통 feature를 고정할 때의 직접 gradient 분리가 실제 구현에서 확인됐다.
Shared encoder/embedding을 통한 영향까지 사라진다는 뜻은 아니다.
집단당 rank가 32에서 16으로 줄기 때문에 v1과의 차이를 순수한 sharing
효과 하나로 인과적으로 분해할 수도 없다.

CPU gate는 원래 계약 그대로 64개 train entity, 32개 validation entity,
25 epochs로 두 번 독립 실행했다. 학습 이력과 최적 tensor가 동일했고,
train objective는 **48.3266%** 감소했다. 생성 support/vocabulary/finite
value/zero-gap 검사도 통과했다. 최적 validation epoch는 두 번 모두 0으로,
작은 집합의 과적합 검증이며 일반화 개선을 입증하는 결과는 아니다.

## 실행 범위

Source commit은 CPU와 GPU 모두
`c77d2f9558ca019d14ccf4b833cbc6628728ff44`이다. 실행 전에 커밋했다.
등록된 8개 데이터 view와 train-only oracle/cache를 그대로 재사용했고,
기존 prepared index의 SHA 및 각 입력 cache SHA를 검사했다.

이번에는 pi=0.05의 kappa=0/1에서 두 후보씩 **GPU 학습 4건**을 수행했다.
각 조건은 train 31,951개 entity(희소 집단 1,545개), validation 6,846개
entity(희소 집단 331개)다. 학습 시드 20260930, FP32, AdamW lr=.001,
weight decay=1e-5, batch=512, clip=1, 최대 50 epochs, patience=5,
scheduler 없음, checkpoint 선택은 global validation base NLL이다.

실제 조기 중단은 15–16 epochs, 선택 checkpoint는 0-based epoch 9–10이었다.
RTX 3090 네 장을 사용했다. 다른 작업의 GPU 부하도 있었으므로 기록한
wall time을 모델 속도 비교로 해석하지 않는다.

Validation은 checkpoint 선택과 사전 지정한 반응 audit에 사용했다.
이미 사용한 development data를 재사용하는 탐색적 파일럿이며,
새로운 독립 확증 결과는 아니다. Held-out test는 접근하지 않았다.

## 결과: 같은 5% 조건에서의 v1/v2 비교

모든 수치는 history 내부를 먼저 평균한 뒤 entity 간 평균한 copy 확률
반응 범위다. 모든 validation nonfirst history와 31개 전체 gap bin을
사용했다. 이는 predictive input response이며 causal do(gap) 효과가 아니다.

| 데이터·집단 | v1 CS-B1 | v2 CS2-B1 | v2 판정 |
|---|---:|---:|---|
| kappa=0, label=0: 의존성 없음·다수 집단 | 0.038017 | 0.019987 | null 상한 통과 |
| kappa=0, label=1: 의존성 없음·희소 집단 | 0.045730 | **0.081949** | **null 상한 실패** |
| kappa=1, label=0: 의존성 있음·비활성 집단 | 0.046383 | 0.033930 | null 상한 통과 |
| kappa=1, label=1: 의존성 있음·활성 집단 | 0.356849 | 0.343783 | 활성 반응 기준 통과 |

실패 셀의 관측 repeat 반응도 **0.080610**으로 상한 0.05를 넘었다.
활성 copy/repeat 반응은 각각 0.343783/0.338277이며, 활성/최대 null
비율 >=2 조건은 둘 다 통과했다. 실패 항목은 **copy와 repeat의 null
반응 상한**이다. Zero-gap 및 생성 유효성 기준은 통과했다.

같은 v2 구조를 쓰는 일반 목적함수 대조군도 보존했다.

| 데이터·집단 | CS2-U1 copy | CS2-B1 copy | CS2-U1 repeat | CS2-B1 repeat |
|---|---:|---:|---:|---:|
| kappa=0, label=0 | 0.020427 | 0.019987 | 0.020103 | 0.019661 |
| kappa=0, label=1 | **0.055189** | **0.081949** | **0.054313** | **0.080610** |
| kappa=1, label=0 | 0.025898 | 0.033930 | 0.025490 | 0.033385 |
| kappa=1, label=1 | 0.365049 | 0.343783 | 0.359321 | 0.338277 |

CS2-U1도 의존성 없는 희소 집단에서 0.05를 초과했다. 따라서 실패를
balanced 보조 손실만의 문제로 귀속시킬 수 없다. 다만 동일 초기화와
등록된 학습 절차에서 B1의 해당 반응이 U1보다 더 컸다는 관측은 남는다.
U1의 활성 반응이 더 크다는 사실 역시 U1의 정확도가 더 높다는 증거는 아니다.

| 모델·데이터 | 실행 epochs | 선택 epoch (0-based) | 최적 validation base NLL |
|---|---:|---:|---:|
| CS2-U1, kappa=0 | 16 | 10 | 8.219602 |
| CS2-B1, kappa=0 | 15 | 9 | 8.221156 |
| CS2-U1, kappa=1 | 16 | 10 | 8.194628 |
| CS2-B1, kappa=1 | 16 | 10 | 8.195608 |

모델당 2,048개, 총 **8,192개 entity**를 같은 train-only parent/length
계획으로 생성했다. Nonfirst gap **189,324개**의 support 위반은 0,
예약 mark 출력은 0, numeric value는 모두 유한했다. 동일 가중치의
zero-gap control 최대 반응도 모두 정확히 0이었다. 이 support는 empirical
train atoms이며 연속 DGP의 population support 전체를 의미하지 않는다.

## 판정과 연구상 의미

1. 등록한 구조 변경은 구현됐고, 의도한 직접 gradient 분리도 검증됐다.
   따라서 이번 결과를 단순히 "경로를 제대로 분리하지 못했다"고 설명할
   근거는 없다. 다만 모든 가능한 구현 오류를 배제했다는 주장은 아니다.
2. 활성 데이터의 다수 비활성 집단 반응 감소는 관측됐으나, 다른 null
   조건에서 더 큰 문제가 발생했다. **완전한 경로 분리만으로 null safety를
   확보한다는 수정 가설은 이 등록 파일럿을 통과하지 못했다.**
3. 새 실패는 kappa=0에서도 발생한다. 이 데이터에는 진짜 활성 집단이
   없으므로, 현상을 "진짜 활성 신호의 전달" 하나로 설명할 수 없다.
4. 두 objective 모두 희소 집단에서 실패한 결과는 공유 제약을 줄였을 때
   집단 내부의 불필요한 반응을 학습할 가능성과 양립한다. 그러나 희소
   표본 분산, 공유 feature, rank 변경, 초기화, 최적화 중 어느 것이 얼마나
   기여했는지는 이번 한 시드 결과로 분해할 수 없다.
5. Pilot 전체는 FAIL이다. 부모 runner는 정상적인 scientific FAIL 코드
   **2**로 종료했고 worker의 `FAILED.json`은 없다. Pi=.10/.25/.50,
   five-seed, real-data, held-out 실험을 실행하지 않았다. 대조군으로
   primary를 바꾸거나 임계값을 완화하지 않았다.

## 다음에 구분해야 할 질문

다음 목표는 **새 구조를 계속 추가하기 전에, 목적함수의 추가 비중과 집단
균형화 효과를 분리하는 것**이다. 현재 U1/B1 비교는 이 두 효과를 동시에
바꾸며, ordinary U1 자체의 희소 null 실패도 설명해야 한다.

후속 대조실험으로는 같은 v2 구조에 다음 세 목적함수를 비교할 수 있다.
이는 **제안이며 별도 사전등록·구현·실행 전**이다.

```text
U: L_base
A: L_base + mean_repeat_BCE_over_all_train_transitions
B: L_base + mean_over_contexts(mean_repeat_BCE_in_context)
```

현재 U와 B 결과는 보존돼 있다. A는 집단 균형화 없이 repeat 보조 손실만
더하는 대조군이다. Global 집계에서 direct-route의 집단별 계수는
A에서 `T_y/E + T_y/T`, B에서 `T_y/E + 1/2`이고, 양쪽 합은 모두
`T/E + 1`이다. 따라서 A/B 비교는 전체 direct-route 손실 계수를 맞춘
상태에서 집단별 배분 효과를 구분할 수 있다. 이는 AdamW update 크기를
완전히 통제한다는 뜻은 아니며 shared-feature 효과도 포함한다.

이 진단으로도 U의 null 실패가 해결되는 것은 아니다. 결과가 뒷받침할 때
부분 공유나 집단별 잔차의 수축 같은 수정 방향을 검토해야 하며, 그 성공을
미리 가정하지 않는다. 본 v2 등록은 종료됐으므로 새 후보를 같은 파일럿의
fallback으로 실행하지 않는다. 이후 정확도 입증에는 기존과 같이
intervention-error/conditional-TV 집계 계약 및 train-only noninferiority
calibration 재감사가 필요하다.

## 보존 및 재현

- 등록 commit: `d8302e363d2d019704bf050911024dd5bb0b9858`.
- 실행 source: `c77d2f9558ca019d14ccf4b833cbc6628728ff44`.
- 불변 v2 YAML SHA: `057cc59296961e5763b0ee2877fd930bafcab312d6a6637431302424f27f0359`.
- 실행 Python: `/home/finx_sbk/.conda/envs/cofseq/bin/python3`;
  PyTorch `2.1.2+cu121`.
- CPU root: `artifacts/cs_saf/revision_v2/cpu_gate_v1/`.
- GPU root: `artifacts/cs_saf/revision_v2/pilot_v1/`.

각 GPU job의 최적 checkpoint, 전체 epoch 이력, training report,
intervention audit, 생성 sample 및 완료 marker를 원격에 보존했다.
기록 시 **21개 artifact checksum 비교**와 checkpoint tensor/sample 모델
identity를 확인했고, 저장된 생성 sample의 support·vocabulary·유한 값도
다시 검사했다. 원래 v1/forensic artifact는 수정하지 않았다.
Git에는 코드·테스트·설정·작은 결과 문서를 보관하며 큰 runtime artifact는
원격 워크스테이션에 남는다.

재현은 기록한 실행 source의 clean checkout에서 새 output 경로로 한다:

```bash
python -m scripts.run_cs_saf_pilot cpu --revision v2 \
  --cache-root artifacts/cs_saf/prepared_v1 \
  --output artifacts/cs_saf/revision_v2/cpu_reproduction
python -m scripts.run_cs_saf_pilot pilot --revision v2 \
  --cache-root artifacts/cs_saf/prepared_v1 \
  --cpu-gate artifacts/cs_saf/revision_v2/cpu_reproduction/COMPLETE.json \
  --output artifacts/cs_saf/revision_v2/pilot_reproduction --gpus 0,1,2,3
```

새 재현 실행을 수행한 것은 아니다. GPU 간 bitwise 재현을 주장하지 않는다.
YAML의 `PREREGISTERED_NOT_IMPLEMENTED_OR_TRAINED`는 등록 당시의 불변
metadata이며, 최신 상태는 [STATUS.md](STATUS.md)와 이 결과 기록을 따른다.
