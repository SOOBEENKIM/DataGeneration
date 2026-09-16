# CS-SAF v3 explicit-history/residual pilot — 2026-09-16

**사전등록한 v3 수정과 원격 실행을 완료했다.** 아래 판정은 pi=.05, 단일 seed의
탐색적 파일럿이며 최종 방법 성공이나 외부 baseline 우위가 아니다.

- 등록 `9813573`, 첫 구현 `c9c516a`, 최종 CPU/GPU source `44a2bd3848abd1fdddf0fc37d2cd6894109e3d65`.
- **새 GPU 학습 8개**: H/E/C/R × kappa 0/1. 기존 U는 가중치를 고정해 동일 평가만 수행.
- [등록 계약](revision_v3_preregistration.md) / [불변 YAML](../../configs/benchmark_v2/cs_saf_revision_v3.yaml)
  / [전체 수치·해시](v3_pilot_v1_result.json).
- 모델 seed 20260930, 최대 50 epoch, patience 5, 동일 초기 공통 가중치/배치 순서,
  global validation base NLL로 best 선택. best와 epoch 9 모두 보존·평가했다.
- 후속 prevalence, 추가 seed, 외부 모델 학습, held-out 접근 없음.

## 수정과 통제

기존 context별 rank-16 feature `u_s(c)`에 독립적인 이력 보정 계수 alpha[2,16]를
추가했다. `h_s(c)=dot(alpha_s,u_s(c))/sqrt(16)`, 총 **32개 parameter 추가**다.
현재 gap 경로 r는 train 기준 평균을 뺀 `delta=r-E_pi[r]`로 분리한다.

| 모델 | Copy logit | 역할 |
|---|---|---|
| 기존 U | b+r | 이전 ordinary-loss checkpoint |
| H | b+h | 비선형 이력 전용, current-gap 없음 |
| E | b+h+r | 이력 보정 추가 효과 |
| C | b+h+delta | 동일 용량에서 중심화/최적화 좌표 효과 |
| R | b+h+delta | C와 같은 구조에 잔차 규제만 추가 |

H/E/C/R는 모두 133,581개 parameter를 저장한다. H에는 1,056개 dormant gap-bank
parameter가 있으므로 active 용량까지 같은 모델이라고 주장하지 않는다.
E/C/R의 active parameter 수는 동일하다. 이력·잔차는 계수는 분리하지만 GRU와
u feature는 공유하므로 완전한 feature/gradient 독립을 주장하지 않는다.

R만 `base + .01 * mean_transition[sqrt(E_pi[delta^2]+1e-8)-1e-4]`를 최소화한다.
수치적으로 같은 유리화 식을 사용해 0 부근 상쇄 오차를 피한다. 관측 train의
context별 gap-bin 빈도를 고정하며, 첫 사건/padding을 제외한 고정 train 분모를 쓴다.
규제 강도는 하나로 고정했으며 사후 sweep/조정은 없다. 보조 repeat BCE와 balanced
loss도 사용하지 않았다. 어느 context가 활성인지 학습에 알려주지 않았다.

중심화 전후 함수는 alpha의 대응 변환으로 같게 만들 수 있다. 그러므로 C의 변화는
새 표현력의 증명이 아니며, R−C가 동일 구조에서 규제 효과를 보는 핵심 비교다.
E−U는 explicit history head를 추가한 비교다. 단일 최적화 trajectory와 GPU 수치
차이까지 포함하므로 보편적인 구조 인과 효과로 일반화하지 않는다.
또한 E/C의 대응 변환에 맞춰 Adam 상태나 weight decay를 변환하지 않았다.
같은 함수 계열이라는 사실이 동일한 최적화나 동일한 함수 공간의 규제를 뜻하지 않는다.
C−E는 이 parameterization 효과를 함께 포함하며, 중심화가 일반적으로 해롭다는 증거는 아니다.

## Best validation: 반응과 관측 분포 정확도를 같이 평가

TV는 낮을수록 좋다. 각 관측 이력에서 64개 mark 전체 분포를 DGP의 관측정보 기반
oracle과 비교하고, train의 context별 주변 gap-bin 빈도로 적분한 값이다.
이력 내부 평균 후 entity 평균, null 요약은 세 null cell의 동일 가중 평균이다.
이것은 history-conditional gap density 적분이나 free-running 생성 분포 오차가 아니다.

| 모델 | 가장 큰 null copy range | active copy range | 세 null 평균 mark TV | active mark TV | 반응 gate |
|---|---:|---:|---:|---:|---|
| CS2-U1 | 0.055188959 | 0.365049046 | 0.052583982 | 0.086640340 | historical FAIL |
| CS3-H1 | 0.000000000 | 0.000000000 | 0.052170955 | 0.139144228 | FAIL |
| CS3-E1 | 0.056817306 | 0.380408746 | 0.052847143 | 0.085479746 | FAIL |
| CS3-C1 | 0.058481466 | 0.380482101 | 0.052865910 | 0.102974418 | FAIL |
| CS3-R1 | 0.003943601 | 0.340098053 | 0.052470000 | 0.096924737 | PASS |

H의 0 반응은 구조적 음성 대조군의 성질이며 active gate를 통과할 모델이 아니다.
원래 copy/repeat 양쪽의 active≥.05, 모든 null≤.05, selectivity≥2,
zero-gap invariant 및 support/출력 유효성 기준을 그대로 적용했다.

추가 정확도 screen은 R−C와 R−U 각각에서 null 평균 TV<0, active TV≤0이어야 한다.
이는 사전등록한 보수적인 탐색적 방향 확인이며 유의성 또는 보정된 비열등성 검정이 아니다.

| 비교 | 세 null 평균 TV 차이 | active TV 차이 | 정확도 screen |
|---|---:|---:|---|
| R − CS3-C1 | -0.000395910 | -0.006049681 | PASS |
| R − CS2-U1 | -0.000113982 | +0.010284397 | FAIL |

**Primary R 반응 gate: PASS. 현재 R 후보의 후속 확장 자격: FAIL.**

1. **이번 pi=.05 단일 seed에서는 기존 반응 실패를 해소했다.** R의 가장 큰 null
   copy range는 .003943601, active는 .340098053이다. Observable repeat도 null
   최대 .003880979, active .334719523으로 모든 원래 반응 조건을 통과했다.
   이는 여러 prevalence/seed의 null-safety 보장으로 확장할 수 없다.
2. **이력 보정 추가와 중심화만으로는 부족했다.** E/C의 희소 null copy range는
   .056817306/.058481466으로 .05를 넘는다. H는 null에서 유용한 예측을 하지만
   active TV가 .139144228로 나쁘다. 현재 gap 정보를 모두 지우는 해법은 적절하지 않다.
3. **규제 효과 자체는 C 대비 양쪽에서 좋았다.** R−C는 세 null cell TV 모두 개선,
   active TV도 −.006049681 개선했다. 단순히 규제가 활성 신호를 파괴한 것이
   이번 결과의 유일한 설명은 아니다.
4. **현재 구조는 기존 U의 활성 정확도를 회복하지 못했다.** Active TV는 U .086640340,
   E .085479746, C .102974418, R .096924737이다. E→C 악화 +.017494672는
   동일 용량/함수 계열에서도 중심화 좌표와 학습 경로가 중요함을 보여주는 단일-seed
   관찰이다. R이 그 악화를 일부 줄였지만 U보다 +.010284397 나쁘다.
5. **이 실패는 grid 가중치 하나만의 문제도 아니다.** Active factual-bin TV도
   U .079378825 → R .090397077, repeat BCE도 .451531453 → .457321257로 악화됐다.
   반대로 R의 세 null 평균 TV 이득은 U 대비 −.000113982로 작으며,
   세 null 중 두 cell은 U보다 약간 나쁘다. null 전체 정확도 우위를 일반화하지 않는다.
6. **고정 epoch 9에서도 중요한 방향은 같다.** Active C−E +.018550066,
   R−C −.007056398, R−U +.001821604이다. Best epoch만 골라 생긴 부호는 아니다.
   다만 같은 학습 trajectory의 두 snapshot이므로 독립 seed 재현은 아니다.

따라서 결과는 **반응 기준 PASS / 기존 일반 모델 대비 정확도 기준 FAIL**로 보존한다.
규제 강도를 사후 변경하거나 active 기준을 낮춰 성공으로 재분류하지 않았다.

## 개별 context와 fixed-epoch 결과

kappa=0은 두 context 모두 gap-null, kappa=1은 label=1만 active다.
copy range가 곧 실제 latent copy 확률의 정확도는 아니다. Learned fresh head가
uniform이 아닐 수 있어, 내부 q뿐 아니라 전체 mark TV를 직접 계산했다.

| 모델 | kappa | label | grid mark TV | factual-bin TV | grid repeat L1 | factual repeat BCE | copy range | repeat range |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CS2-U1 | 0 | 0 | 0.048907213 | 0.048872903 | 0.031617880 | 0.492823250 | 0.020427480 | 0.020102930 |
| CS2-U1 | 0 | 1 | 0.057418531 | 0.057342379 | 0.038961223 | 0.489598540 | 0.055188959 | 0.054313086 |
| CS2-U1 | 1 | 0 | 0.051426202 | 0.051407848 | 0.033743709 | 0.495194713 | 0.025898499 | 0.025489811 |
| CS2-U1 | 1 | 1 | 0.086640340 | 0.079378825 | 0.070245223 | 0.451531453 | 0.365049046 | 0.359321445 |
| CS3-H1 | 0 | 0 | 0.049252388 | 0.049252388 | 0.031938883 | 0.492981578 | 0.000000000 | 0.000000000 |
| CS3-H1 | 0 | 1 | 0.056246214 | 0.056246214 | 0.037278244 | 0.488514466 | 0.000000000 | 0.000000000 |
| CS3-H1 | 1 | 0 | 0.051014262 | 0.051014262 | 0.035831029 | 0.495739981 | 0.000000000 | 0.000000000 |
| CS3-H1 | 1 | 1 | 0.139144228 | 0.119747066 | 0.129306949 | 0.484568766 | 0.000000000 | 0.000000000 |
| CS3-E1 | 0 | 0 | 0.048736230 | 0.048726336 | 0.031345550 | 0.492827400 | 0.015238596 | 0.014996734 |
| CS3-E1 | 0 | 1 | 0.058169129 | 0.058068555 | 0.039885419 | 0.489861049 | 0.056817306 | 0.055916174 |
| CS3-E1 | 1 | 0 | 0.051636071 | 0.051639889 | 0.036657017 | 0.496017995 | 0.011459829 | 0.011277743 |
| CS3-E1 | 1 | 1 | 0.085479746 | 0.075118544 | 0.070138775 | 0.449203949 | 0.380408746 | 0.374398227 |
| CS3-C1 | 0 | 0 | 0.049327259 | 0.049319434 | 0.032024242 | 0.493010828 | 0.011613261 | 0.011428649 |
| CS3-C1 | 0 | 1 | 0.057150450 | 0.056979272 | 0.038431313 | 0.489308128 | 0.058481466 | 0.057552765 |
| CS3-C1 | 1 | 0 | 0.052120019 | 0.052104758 | 0.037049739 | 0.496113368 | 0.010041354 | 0.009881545 |
| CS3-C1 | 1 | 1 | 0.102974418 | 0.096634875 | 0.089708196 | 0.462070333 | 0.380482101 | 0.374457975 |
| CS3-R1 | 0 | 0 | 0.049245654 | 0.049245620 | 0.031926802 | 0.492977960 | 0.000072643 | 0.000071490 |
| CS3-R1 | 0 | 1 | 0.056259528 | 0.056246801 | 0.037292261 | 0.488548424 | 0.003943601 | 0.003880979 |
| CS3-R1 | 1 | 0 | 0.051904817 | 0.051904753 | 0.036810047 | 0.496040923 | 0.000073707 | 0.000072535 |
| CS3-R1 | 1 | 1 | 0.096924737 | 0.090397077 | 0.083261563 | 0.457321257 | 0.340098053 | 0.334719523 |

선택한 best뿐 아니라 고정 epoch-9 결과를 모두 남긴다. 아래는 grid mark TV다.
같은 seed trajectory의 두 checkpoint는 독립 반복이 아니다.

| 모델 | snapshot | split | k0/y0 | k0/y1 | k1/y0 | k1/y1 active |
|---|---|---|---:|---:|---:|---:|
| CS2-U1 | best | train | 0.048675908 | 0.056419651 | 0.051058034 | 0.086712079 |
| CS2-U1 | best | validation | 0.048907213 | 0.057418531 | 0.051426202 | 0.086640340 |
| CS2-U1 | epoch_9 | train | 0.050052538 | 0.058659855 | 0.052064396 | 0.092430646 |
| CS2-U1 | epoch_9 | validation | 0.050292596 | 0.059462242 | 0.052428640 | 0.092125368 |
| CS3-H1 | best | train | 0.049012772 | 0.055242554 | 0.050784977 | 0.138289551 |
| CS3-H1 | best | validation | 0.049252388 | 0.056246214 | 0.051014262 | 0.139144228 |
| CS3-H1 | epoch_9 | train | 0.050357867 | 0.057406210 | 0.051394946 | 0.138749936 |
| CS3-H1 | epoch_9 | validation | 0.050633415 | 0.058237472 | 0.051754251 | 0.139648249 |
| CS3-E1 | best | train | 0.048502076 | 0.057064993 | 0.051418447 | 0.085544716 |
| CS3-E1 | best | validation | 0.048736230 | 0.058169129 | 0.051636071 | 0.085479746 |
| CS3-E1 | epoch_9 | train | 0.050175167 | 0.059498183 | 0.051957643 | 0.082458836 |
| CS3-E1 | epoch_9 | validation | 0.050413319 | 0.060358669 | 0.052309238 | 0.082453305 |
| CS3-C1 | best | train | 0.049086361 | 0.056124933 | 0.051903865 | 0.103510673 |
| CS3-C1 | best | validation | 0.049327259 | 0.057150450 | 0.052120019 | 0.102974418 |
| CS3-C1 | epoch_9 | train | 0.050439505 | 0.058190155 | 0.052742596 | 0.101590661 |
| CS3-C1 | epoch_9 | validation | 0.050716270 | 0.059008707 | 0.053082560 | 0.101003371 |
| CS3-R1 | best | train | 0.049005901 | 0.055251175 | 0.051684743 | 0.097290810 |
| CS3-R1 | best | validation | 0.049245654 | 0.056259528 | 0.051904817 | 0.096924737 |
| CS3-R1 | epoch_9 | train | 0.050356058 | 0.057414497 | 0.052440045 | 0.094328583 |
| CS3-R1 | epoch_9 | validation | 0.050631899 | 0.058247665 | 0.052788665 | 0.093946972 |

각 paired contrast의 entity SE/중앙값/p90은 evidence JSON에 있다.
entity SE를 seed 불확실성으로 해석하거나 반복 사용한 validation으로 확증적
유의성을 주장하지 않는다. 평가 oracle의 latent state 정답은 로드하지 않았다.

## 검증, 수치 수정, 보존

- 관련 테스트 **106 PASS**: 함수 보존, centering, penalty gradient, fixed denominator,
  첫 사건/padding, strict past, context permutation, 전체 mark TV, batch 정렬 등.
- 같은 source에서 CPU 두 번씩, 네 후보 모두 PASS. Train objective 감소 약 48%.
- 첫 CPU 결과는 상수 logit을 펼친 뒤 sigmoid를 계산할 때 최대 5.9604645e-8의
  가짜 range가 나와 fixed 1e-8 gate에서 FAIL했다. 같은 저장 상태에서 logit range=0,
  펼친 sigmoid range>0, 한 번 계산 후 broadcast range=0을 직접 재현했다.
- 상수 확률을 한 번만 계산하도록 바꿨고, **수정 전후 CPU best 가중치가 모두 동일**하다.
  기준·손실·규제·예산은 바꾸지 않았다. 첫 CPU 실행도 보존했다.
- 최종 검증: 파일 해시 97회,
  checkpoint tensor 20회,
  entity 배열 집단 80개,
  요약 통계 2240개 재계산.
- 생성 16,384 entity /
  378,648 nonfirst gap. 각 생성 파일의
  model-state identity와 support/mark/value 검사를 실행했다.

[모델](../../models/cs_saf_v3.py), [학습/평가](../../experiments/cs_saf_v3.py),
[실행기](../../scripts/run_cs_saf_v3.py), [테스트](../../tests/test_cs_saf_v3.py).
원격 artifact root는 `artifacts/cs_saf/revision_v3/`이며 `cpu_v1`(실패 보존),
`cpu_v2`(PASS), `gpu_v1`(전체 결과)을 사용했다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/finx_sbk/.conda/envs/cofseq/bin/python3 \
  -m scripts.run_cs_saf_v3 cpu --cache-root artifacts/cs_saf/prepared_v1 \
  --output artifacts/cs_saf/revision_v3/cpu_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/finx_sbk/.conda/envs/cofseq/bin/python3 \
  -m scripts.run_cs_saf_v3 run --cache-root artifacts/cs_saf/prepared_v1 \
  --cpu-gate artifacts/cs_saf/revision_v3/cpu_v2/COMPLETE.json \
  --output artifacts/cs_saf/revision_v3/gpu_v1 --gpu 2
```

실행 source의 clean checkout과 기존 cache가 필요하다. 재실행은 새 output 폴더를
사용해야 하며 기존 artifact를 덮어쓰지 않는다. 코드·계약·요약 근거는 Git에,
큰 checkpoint/샘플/배열은 워크스테이션에 보존한다.

## 연구 기여의 경계와 다음 단계

다음에 분리해 검증할 가설은 **중심화를 forward parameterization에 강제할 필요가
있는가, 아니면 규제할 성분을 정의하는 데만 사용해도 되는가**다. 구체적인 후보는
E의 `b+h+r` forward를 유지하고, R과 같은 `rho(r−E_pi[r])` penalty를 적용하는 것이다.
현재 비교의 미실행 조합인 “비중심화 forward + 중심화한 성분 규제”를 채워,
표현 용량·규제 강도를 추가로 바꾸지 않고 parameterization과 penalty의 결합을 본다.

이것은 **후속 제안이며 아직 등록·구현·학습하지 않았다.** 이번 등록의 실패 후
fallback 금지와 일치하게 자동 실행하지 않았다. 새 계약은 E/R/U 비교, 동일 초기값과
예산, 양쪽 response/accuracy 기준, 중단 규칙을 다시 고정해야 한다. R−C가 active
정확도도 개선했으므로 근거 없이 lambda만 낮추는 탐색을 우선하지 않는다.

이후 선택성과 정확도가 함께 통과해야 더 넓은 prevalence, 여러 seed, 실제 순차
baseline과 실데이터 비교로 갈 근거가 생긴다. 현재는 희소화에 따른 dilution,
외부 모델 우위, privacy/utility 우위 또는 논문 기여 확정의 증거가 아니다.

기존 [선행연구/주장 감사](research_claims_and_related_work_audit_2026_09_16.md)의
copy mixture, centering, norm regularization, behavioral evaluation 관련 독창성
제약은 유지된다. 이번에 새로운 것은 등록한 구조의 실제 학습 결과와 대조군 비교
증거이며, 구성 요소 자체를 처음 제안했다는 주장은 하지 않는다.
