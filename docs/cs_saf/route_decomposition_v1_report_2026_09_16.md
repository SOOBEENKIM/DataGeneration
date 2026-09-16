# Fixed-checkpoint history/gap decomposition: result, 2026-09-16

**완료: 이력 보정을 보존한 진단은 원인을 좁혔지만, 새 모델의 학습 성공은 아니다.**
일반 손실 U의 비활성 희소 집단에서 route의 평균 성분은 유용하다.
그 평균은 남기고 gap 변화만 제거하면 검증 손실은 작게 개선되지만 학습 손실은 악화된다.
활성 집단에서는 같은 제거가 학습·검증 모두 크게 악화시킨다.
따라서 유용한 이력 보정과 gap 변화를 분리해서 다룰 근거는 생겼지만,
“비활성 residual이 학습·검증 모두 해롭다”는 사전등록 가설은 충족하지 않았다.

근거: [사전등록](route_decomposition_v1_preregistration.md),
[고정 YAML](../../configs/benchmark_v2/cs_saf_route_decomposition_v1.yaml),
[전체 수치·해시](route_decomposition_v1_result.json),
[모델·이론·전처리·선행연구·주장 감사](research_claims_and_related_work_audit_2026_09_16.md).

## 범위와 실행 이력

- 사전등록 `d12021b`, 구현 `ad01aaf`, 최종 실행 source `7d5a1be64f198f9cc56fa1a9d085460a704bbad8`.
- 기존 pi=.05, kappa=0/1, U/A/B의 **6개 학습 실행에서 best/epoch-9 12개 snapshot 기록**.
  서로 다른 seed 12개가 아니다. B의 kappa=0 best와 epoch-9는 같은 상태다.
- 전체 train 31,951개 entity / 734,984개 transition, validation 6,846개 / 157,271개.
  희소 entity는 각각 1,545 / 331개. 양쪽 observed context와 31개 support bin 전부 포함.
- **새 학습 0, optimizer step 0, 생성 표본 0, held-out test 접근 0**.
  이미 checkpoint 선택/진단에 사용한 validation을 다시 분석한 탐색적 진단이다.
- 관련 테스트 **92 PASS**. 같은 source의 CPU 검증 PASS 후 GPU 0/1/2에서 실행 완료.
- 초기 `audit_v1`은 GPU 실행 batch 128이 부모 분석의 256과 달라 historical BCE
  비교 오차가 1.5671924e-6까지 생겨 중단됐다. 고정 허용오차 1e-6를 완화하지 않았다.
  같은 checkpoint의 통제 비교에서 batch 256의 오차는 6.1914e-10 이하였다.
  배치를 부모와 일치시킨 뒤 `cpu_v2` / `audit_v2`에 재실행했다.
  실패 terminal과 수치 비교는 그대로 보존했고 evidence JSON에도 넣었다.
  이 통제 비교는 배치 형태 의존성을 확인했으며 특정 CUDA kernel 하나를 고립한 것은 아니다.

## 무엇을 분리했는가

현재 copy logit은 `b(c)+r_s(c,k)`다. c는 과거 이력+observed static 정보,
k는 현재 gap bin이다. train에서만 계산한 context별 gap 빈도 pi_s를 고정한다.

```text
mu_s(c) = sum_k pi_s(k) r_s(c,k)
delta_s(c,k) = r_s(c,k) - mu_s(c)
F: b + mu + delta       원 모델
M: b + mu               평균 이력 보정만 유지
R: b + delta            평균 보정만 제거
Z: b                    경로 전체 제거
q = sigmoid(copy_logit)
p_repeat = q + (1-q) p_fresh(previous_mark)
```

logit에서 분해한 후 sigmoid를 적용한다. 확률을 평균하는 다른 실험과 혼동하지 않는다.
pi_s는 context별 주변 빈도이며 `p(gap|history)`가 아니다. mu는 이 기준에서
현재 gap과 무관한 함수이지, 실제 이력 메커니즘을 유일하게 식별한 결과가 아니다.
이 기준 의존성은 [Lengerich et al. (2020)](https://proceedings.mlr.press/v108/lengerich20a.html)의
주효과/상호작용 식별 문제와 관련된다. 이번 분석은 완전한 functional-ANOVA가 아니다.

M/Z의 gap 반응은 **활성 집단에서도 수학적으로 0**이다. 이것을 null-safety를
학습한 성공으로 세지 않는다. 실제 null 집단에만 M, active 집단에만 F를 적용하는
oracle 선택도 배포 가능한 해결책으로 주장하지 않는다.

## 결과: best checkpoint의 validation repeat BCE

낮을수록 좋다. 각 entity 내부 transition 평균 후 context 내부 entity 평균이다.
아래는 repeat 여부의 사실적 예측 손실이며 free-running 생성 품질 평가가 아니다.
kappa=0에서는 두 집단 모두 gap-null, kappa=1에서는 label=1만 active다.
U=일반 손실, A=전역 평균 보조 손실, B=집단 균형 보조 손실.
각 kappa/목적함수에는 각각 학습한 checkpoint를 사용한다.

| kappa | 목적함수 | label | F 원 모델 | M 평균만 | R residual만 | Z 경로 제거 | M−F |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | U | 0 | 0.492823250 | 0.492770121 | 0.540908037 | 0.540856977 | -0.000053129 |
| 0 | U | 1 | 0.489598539 | 0.488799468 | 0.542980485 | 0.542022948 | -0.000799070 |
| 0 | A | 0 | 0.491865451 | 0.491792160 | 0.527456477 | 0.527364110 | -0.000073291 |
| 0 | A | 1 | 0.488559919 | 0.487643023 | 0.528073561 | 0.526989718 | -0.000916896 |
| 0 | B | 0 | 0.492587740 | 0.492521257 | 0.509992972 | 0.509923092 | -0.000066484 |
| 0 | B | 1 | 0.495011023 | 0.493498217 | 0.506823586 | 0.505112630 | -0.001512806 |
| 1 | U | 0 | 0.495194713 | 0.495023652 | 0.520918222 | 0.520759821 | -0.000171061 |
| 1 | U | 1 | 0.451531454 | 0.503302035 | 0.455516693 | 0.506302850 | 0.051770582 |
| 1 | A | 0 | 0.494403313 | 0.494246160 | 0.510052907 | 0.509852208 | -0.000157152 |
| 1 | A | 1 | 0.453562193 | 0.503842555 | 0.455227063 | 0.503746502 | 0.050280362 |
| 1 | B | 0 | 0.495405469 | 0.495197487 | 0.509697317 | 0.509560850 | -0.000207981 |
| 1 | B | 1 | 0.447481940 | 0.492483512 | 0.447817727 | 0.496357381 | 0.045001572 |

U의 주 분석인 kappa=0/label=1에서:

- 전체 경로를 끄면 `.489598539 → .542022948`: 유용한 보정을 함께 잃는다.
- 평균만 보존하면 `.488799468`: 경로 전체 제거보다 `.053223480` 좋아진다.
- gap residual만 제거한 개선은 `.000799070`으로 훨씬 작다.
- 평균만 제거한 R은 `.542980485`로 나빠진다. 고정된 나머지 가중치 아래에서
  큰 이득은 평균 성분에 있고, gap 변동의 validation 이득은 없다는 해석이다.
- 같은 U의 active validation에서 F→M은 `.451531454 → .503302035`,
  **+.051770582 악화**다. 유용한 gap 의존성까지 일괄 제거할 수 없다.

A/B도 희소 null에서 M−F는 train 양수, validation 음수이고, active에서 모두 양수다.
B의 null validation 개선 폭은 U보다 크지만, M으로 바꿔도 B `.493498217`은
U 원 모델 `.489598539`보다 나쁘다. 따라서 residual 제거만으로 목적함수 간
차이까지 해소되는 것은 아니다. 세 목적함수의 우열 일반화도 불가하다.

## 사전등록 가설을 그대로 판정

주 분석 U/희소 집단의 paired contrasts. SE는 고정 모델 아래 entity 차이의
기술 통계로, seed 불확실성이나 독립 재현성, 확증적 유의성을 뜻하지 않는다.

| kappa | snapshot | split | M−Z | M−F | M−F entity SE |
|---|---|---|---:|---:|---:|
| 0 | best | train | -0.053433778 | +0.000829508 | 0.000155910 |
| 0 | best | validation | -0.053223480 | -0.000799070 | 0.000318914 |
| 0 | epoch_9 | train | -0.048606835 | +0.000755713 | 0.000152141 |
| 0 | epoch_9 | validation | -0.047911829 | -0.000790743 | 0.000312096 |
| 1 | best | train | -0.002587532 | +0.058147048 | 0.001719175 |
| 1 | best | validation | -0.003000815 | +0.051770582 | 0.003685246 |
| 1 | epoch_9 | train | -0.001430503 | +0.056999859 | 0.001640791 |
| 1 | epoch_9 | validation | -0.001794555 | +0.050807939 | 0.003487683 |

1. **평균 성분 유용성: 방향 일치.** kappa=0의 train/validation, best/epoch-9 모두 M−Z<0.
2. **비활성 residual의 양쪽 split 유해성: 조건 미충족.** validation에서는 제거가
   좋지만 train에서는 나쁘다. validation만 골라 성공으로 재정의하지 않는다.
3. **활성 residual 유용성: 방향 일치.** kappa=1의 네 비교 모두 M−F>0.

이 패턴은 비활성 집단의 gap 변화가 표본 내 fitting에는 기여하지만 validation에는
도움이 되지 않는다는 해석과 맞는다. 과적합 가능성을 지지하는 관찰이지만, 학습 과정의
유일한 원인, 보편적인 regularization 효과, 새 구조의 성공을 증명하지 않는다.
checkpoint 두 개는 같은 trajectory이며 validation은 반복 사용되었다.

## 기계적 검증과 보존

모든 고정 한계를 통과했다. 최대 오차:

| 확인 | 관측 최대 |
|---|---:|
| 원 logit 재구성 | 2.22e-16 |
| 빈도 가중 residual 평균=0 | 3.16e-16 |
| 원 forward copy/repeat 확률과 일치 | 1.24e-7 |
| 이전 F/Z BCE 재현 | 1.37e-9 |
| mark NLL 차이와 repeat BCE 차이 일치 | 1.12e-15 미만 |
| M/Z gap 범위 | 정확히 0 |

23개 파일 해시 비교, 12개 checkpoint tensor 상태 비교, 6개 worker terminal 내용,
48개 entity 배열 집단의 ID 정렬과 4,416개 요약 통계를 재계산해 확인했다.
checkpoint/cache 불변, train/validation entity 분리, train-only tensorizer,
첫 gap 결측, padding/길이, reserved mark 배제, support bin roundtrip을 확인했다.
절대 mark NLL과 repeat BCE는 같지 않다. fresh head를 고정한 **arm 간 차이만**
같다는 likelihood identity를 수치적으로 검증했다.

실행 명령과 entry point:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/finx_sbk/.conda/envs/cofseq/bin/python3 \
  -m scripts.audit_cs_saf_route_decomposition cpu \
  --output artifacts/cs_saf/route_decomposition_v1/cpu_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/finx_sbk/.conda/envs/cofseq/bin/python3 \
  -m scripts.audit_cs_saf_route_decomposition run \
  --cpu-gate artifacts/cs_saf/route_decomposition_v1/cpu_v2/COMPLETE.json \
  --output artifacts/cs_saf/route_decomposition_v1/audit_v2 --gpus 0,1,2
```

실행 source에서만 재현하며 기존 output은 덮어쓸 수 없다. 재실행 시 새 폴더를 사용한다.
[분해 코드](../../experiments/cs_saf_route_decomposition.py),
[실행기](../../scripts/audit_cs_saf_route_decomposition.py),
[검증 테스트](../../tests/test_cs_saf_route_decomposition.py).
checkpoint와 entity 원시 배열은 워크스테이션 runtime 경로에 있고,
Git에는 코드·계약·compact evidence를 저장한다. Git이 전체 checkpoint 백업은 아니다.

## 다음 후보의 근거와 경계

검토할 후보는 **gap-independent history head와 train-centered gap residual을 명시적으로
분리하고, residual만 규제하는 모델**이다. 단순 중심화는 F를 그대로 보존하므로
그 자체로 정확도나 null safety가 개선되지 않는다. 또한 history head의 용량 증가와
regularization 효과가 혼동되지 않도록, 비선형 history-only 대조군과
규제 없는 동일 구조 대조군이 필요하다.

현재 문서는 후보 제안이다. **후속 모델·penalty·lambda·학습 예산은 아직 사전등록하지
않았고 구현·학습하지 않았다.** 다음 계약은 null-label oracle mask 없이 같은 규칙을
양 집단에 적용하고, active 분포 오차와 null 오차를 함께 평가하며, U/A/B 중 어떤
목적함수를 고정할지 정해야 한다. B를 자동 채택할 근거는 없다.
이후에도 고정 pilot → 조건부 분포 정확도 → 여러 seed → 공정한 외부 baseline/실데이터
검증을 거쳐야 한다. 중단된 v1/v2 후속 실험이나 held-out 평가를 재개하지 않았다.
