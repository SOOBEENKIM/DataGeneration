# CS-SAF 구현·CPU 검증·등록 파일럿 결과 — 2026-09-16

**최종 판정: 파일럿 FAIL.** 모델은 활성 집단의 gap 조건부 반응을 학습했지만,
집단 비율 25%에서 CS-B1의 비활성 집단 반응이 고정 상한을 초과했다.
5%·10% 단계는 PASS이며, 중단 규칙에 따라 50% 학습은 실행하지 않았다.
이는 실행 오류가 아니라 사전 등록한 과학적 판정의 실패이다.

원격 워크스테이션에서 구현, 데이터 구성, CPU 검증, GPU 학습을 수행했다.
실행 전에 코드와 설정을 커밋했으며, 이 문서는 실행 이후 결과 기록이다.
전체 수치와 해시는 [pilot_v1_result.json](pilot_v1_result.json)에 저장한다.

## 완성한 모델과 비교 조건

기존 strictly-past GRU(hidden 128), train atom 기반 gap decoder,
copy/new mark factorization, Gaussian value decoder를 유지했다.
Scalar gate를 제거하고 static embedding(8)과 history를 연결한
context에 rank-32 bilinear gap interaction을 구현했다.

```text
past events + static label -> shifted GRU -> history h_t
history h_t               -> train-support gap decoder -> current gap
[history h_t, static] + gap embedding -> bilinear copy logit -> q_t
[history h_t, static]                -> fresh-mark distribution
q_t + fresh-mark distribution        -> current mark
history + current gap + current mark -> numeric value
```

관측 반복과 잠재 copy 선택은 다르므로 보조 손실은 관측 반복 확률
`p_repeat = q + (1-q) * p_new(previous_mark)`에 적용한다.
집단별 전체 train transition 수로 분모를 고정하여, minibatch에 한 집단만
포함되어도 집단 균형 목적함수의 추정량이 유지되도록 했다.
Oracle regime과 oracle 예측값은 모델 학습에 제공하지 않는다.

| 후보 | Gap decoder | Gap/context interaction | 집단 균형 repeat 보조 손실 | 이번 production 학습 |
|---|---|---|---|---|
| CS-C0 | continuous | 없음 | 없음 | 없음; 구현·검사 완료 |
| CS-U0 | train-support aligned | 없음 | 없음 | 없음; 구현·검사 완료 |
| CS-U1 | train-support aligned | 있음 | 없음 | 일반 목적함수 대조군 |
| CS-B0 | train-support aligned | 없음 | 있음 | 없음; 구현·검사 완료 |
| CS-B1 | train-support aligned | 있음 | 있음 | 사전 지정 full candidate |

U0/U1/B0/B1의 state key, shape, 초기값은 동일하다. U1/B1은 각각
133,549개 파라미터를 가진다. C0는 gap head 계열 차이가 있으므로 완전한
파라미터 수 일치를 주장하지 않는다. 구현은 현재 controlled one-label
schema와 최대 32-event sequence용이다.

## 데이터와 정보 경계

비율 0.05/0.10/0.25/0.50과 kappa 0/1의 **8개 데이터 조건**을 구성했다.
기존 entity split을 유지했고, 각 조건은 train 31,951개와 validation
6,846개 entity를 포함한다. Test split의 ID metadata만 유지했으며,
test event/static 본문은 읽거나 새 데이터로 생성하지 않았다.

| 명목 집단 비율 | Train label=1 | Validation label=1 | 활성 train oracle copy 반응 | 모델 학습 |
|---|---:|---:|---:|---|
| 5% | 1,545 | 331 | 0.413742 | 완료 |
| 10% | 3,158 | 653 | 0.411562 | 완료 |
| 25% | 7,972 | 1,659 | 0.411306 | 완료 |
| 50% | 16,053 | 3,421 | 0.411175 | 중단 규칙으로 미실행 |

원래 label uniform을 복원해 nested label을 만들었다. 모든 gap, amount,
timestamp, length, entity ID를 유지했다. Kappa=0에서는 mark도 유지했고,
kappa=1에서는 새로 활성화된 entity의 mark만 원래 DGP 법칙으로 재생성했다.
Train/validation latent regime은 DGP materializer 안에서만 사용했으며,
모델 입력에는 저장하지 않았다. 5% view는 원본 development content와
정확히 같음을 검사했다. 모든 비율의 train-only oracle gate는 PASS이며
null oracle 반응은 모두 0이다.

## CPU와 실행 계약

- 관련 테스트 **54개 PASS**: 모든 후보의 유한 loss/backprop, 미래 정보
  누출 방지, bilinear gradient, 동일 초기값, repeat 확률, 고정 분모,
  데이터 pairing, 생성 표본 저장 및 public API를 검사했다.
- CPU B1을 두 번 독립 실행했다. 64개 train entity, 32개 validation
  entity, 25 epochs이며 학습 이력과 최적 가중치가 정확히 같았다.
  Train objective 감소는 **48.1638%**로 기준 10%를 넘었다.
- 작은 CPU 집합의 최적 validation checkpoint는 epoch 0이었다. 이후
  과적합이 발생했으므로 일반화 개선이 아닌 구현 검증이다.
- GPU 파일럿은 모델 시드 **20260930**, RTX 3090 네 장, FP32,
  AdamW lr=0.001, weight decay=1e-5, batch=512, 최대 50 epochs,
  patience=5, gradient clip=1, scheduler 없음으로 실행했다.
- Checkpoint는 auxiliary를 제외한 global validation base NLL로 선택했다.
  같은 조건의 U1/B1은 초기 가중치와 minibatch 순서, 생성 표본 계획이 같다.
  조기 중단에 따라 실제 학습은 14–16 epochs였다.

전체 설정은 실행 전 커밋한 [실행 계약](pilot_execution_contract_v1.md)과
[pilot YAML](../../configs/benchmark_v2/cs_saf_pilot_v1.yaml)을 따른다.
결과를 본 뒤 loss, seed, epoch, 기준을 바꾸거나 재시도하지 않았다.

## 등록 파일럿 결과

모든 validation nonfirst history에서 train-fitted gap bin 31개를 바꿔
반응 범위를 계산했다. History를 entity 안에서 평균한 뒤 집단별 entity
평균을 사용한다. 이는 **조건부 입력 반응**이며 causal do(gap) 효과가 아니다.
활성 셀은 kappa=1/label=1, null 셀은 나머지 세 조합이다.

CS-B1은 copy와 관측 repeat 각각에서 활성 반응 >=0.05, 모든 null 반응
<=0.05, 활성/최대 null 비율 >=2를 만족해야 한다.

| 비율 | B1 활성 copy | B1 최대 null copy | B1 활성 repeat | B1 최대 null repeat | 판정 |
|---|---:|---:|---:|---:|---|
| 5% | 0.356849 | 0.046383 | 0.351013 | 0.045620 | PASS |
| 10% | 0.364910 | 0.048660 | 0.358871 | 0.047854 | PASS |
| 25% | 0.367779 | **0.051167** | 0.361659 | **0.050313** | **FAIL** |
| 50% | — | — | — | — | 학습 미실행 |

세 비율 모두 최대 null은 **kappa=1/label=0**에서 나타났다. 25%의
kappa=0 null copy 반응은 label 0/1 각각 0.023681/0.026425로 상한보다
작았다. 실패는 활성 데이터 안의 비활성 집단으로 gap 반응이 번진 형태다.
이것은 관측된 실패 위치이며, 원인까지 확정한 것은 아니다.
상한 초과가 작아도 원래 판정을 유지한다. Confidence interval에 따른
사후 면제나 임계값 반올림을 적용하지 않았다.

일반 목적함수 대조군의 결과도 보존한다.

| 비율 | U1 활성 copy | U1 최대 null copy | U1 활성 repeat | U1 최대 null repeat |
|---|---:|---:|---:|---:|
| 5% | 0.269691 | 0.030529 | 0.265435 | 0.030044 |
| 10% | 0.336467 | 0.032844 | 0.331103 | 0.032320 |
| 25% | 0.364233 | 0.045083 | 0.358345 | 0.044352 |

이 시드에서는 B1이 U1보다 활성 반응도 크고 최대 null 반응도 컸다.
반응이 크다는 사실만으로 더 정확한 모델이라고 해석할 수 없다.
사전 지정 primary는 B1이므로 U1로 바꿔 전체 PASS를 선언하지 않는다.

총 **12개 모델 fit**과 모델당 2,048개 entity 생성을 완료했다.
생성된 24,576개 entity의 nonfirst gap 567,972개에서 support violation은
**0**, reserved mark 출력은 **0**, numeric value는 모두 유한했다.
모든 모델에서 동일 가중치를 사용하는 zero-gap control의 최대 반응은
정확히 **0**이었다. Empirical train-atom support 안에 있다는 의미이며,
연속인 DGP population support를 완전히 복원했다는 주장은 아니다.

## 현재 가능한 주장과 한계

1. Oracle 신호 식별 가능성에 더해, 실제 학습한 CS-SAF가 활성 집단의
   gap 조건부 반응을 표현한다는 단일 시드 증거를 얻었다.
2. 현재 full candidate는 세 비율 전체에서 null-safety 기준을 만족하지
   못했다. 따라서 **등록된 CS-SAF v1 파일럿은 실패**다.
3. Oracle train 반응 범위와 learned validation 반응 범위는 데이터와
   목적이 다르며, 두 숫자의 차이를 intervention error로 사용하지 않았다.
4. 집단 비율에 따른 학습 성능 저하, balanced objective의 정확도 개선,
   transition fidelity, worst-context utility, privacy, 비열등성은 아직
   검증하지 않았다. 한 시드의 반응 범위 표로 contribution을 주장할 수 없다.
5. C0/U0/B0 production 비교, 50% 학습, 5-seed confirmatory 실험,
   real-data 비교, held-out 평가를 수행하지 않았다.

## 다음 작업

현재 v1은 여기서 종료한다. 다음 연구 출발점은 **저장된 checkpoint로
null-context 반응의 원인을 분석하고, 근거가 있으면 별도 v2를 등록하는 것**이다.

1. 추가 fit 없이 train history에서 U1/B1의 null 반응을 entity, event
   position, gap bin별로 분해한다. Bilinear logit과 fresh-mark 반복 확률을
   분리해 어느 항에서 차이가 생기는지 확인한다. 목적함수, 공유 표현,
   최적화 중 무엇이 원인인지 현재 결과로 확정할 수 없다.
2. 분석이 뒷받침하는 구조 또는 목적함수 수정 한 가지를 새 버전으로
   사전 등록한다. 어떤 label이 활성인지 DGP 지식을 hard-code하거나,
   oracle target으로 학습하거나, v1의 null 상한을 완화하지 않는다.
3. 수정 후보도 oracle/CPU/동일 pilot gate를 순서대로 통과해야 한다.
   V1의 50% 조건이나 confirmatory 학습을 성공 경로처럼 이어가지 않는다.
4. 이후 confirmatory로 진행하려면 intervention error와 conditional TV의
   집계 단위 및 paired 비교를 고정하고, 기존에 실패했던 train-only
   noninferiority calibration을 먼저 재감사해야 한다. 기존 margin 결과는
   보존하고, 이번 validation 결과에 맞춰 소급 조정하지 않는다.

## 출처와 재현

| 단계 | 실행 source commit |
|---|---|
| 모델·데이터·파일럿 계약, 데이터 materialization | `924131bee7aca609d36ec7e1049aeeeec5801c72` |
| 학습 입력 cache와 비율별 oracle | `43ceef8ef420b4d6517b32110fa23addcc3683fb` |
| CPU gate와 GPU 파일럿 | `0d3be638e180695bf62379af8444398bb0c9708e` |

단계 사이의 수정은 모델 public loss/architecture API 및 생성 표본 보존이다.
Materializer와 cache/oracle 계산 법칙은 바뀌지 않았다. 정확한 provenance는
machine-readable evidence의 전체 commit/hash를 기준으로 한다.
Pilot config SHA-256:
`e36f62998dc537e8359d10b8f0bf706dca35dad8279e0f8a51b5454faff5ac8c`.

원격 worktree:
`/home/ssd-990/soobeenkim/cof-seqgen-0707-2119-Version3-complete/research-cs-saf`.
데이터는 `data/cs_saf/prevalence_v1/`, 입력 cache는
`artifacts/cs_saf/prepared_v1/`, CPU 결과는 `artifacts/cs_saf/cpu_gate_v1/`,
파일럿은 `artifacts/cs_saf/pilot_v1/`에 있다.
각 GPU job에는 `checkpoint_best.pt`, `progress.jsonl`,
`training_report.json`, `intervention_audit.json`, `generated_sample.pt`,
`COMPLETE.json`이 남아 있다. Supervisor 종료 코드는 scientific FAIL을
뜻하는 **2**였고, worker 실행 오류를 뜻하는 `FAILED.json`은 없다.

결과 기록 시 100개 artifact checksum 비교, checkpoint tensor digest와
생성 표본의 모델 identity, source/config/cache identity를 다시 확인했다.
원본 historical worktree와 대형 데이터는 변경하지 않았다.
Git에는 source/config/작은 결과 문서를 보관하며 대형 checkpoint와 데이터는
원격 워크스테이션에 유지한다. Git push는 이 대형 artifact의 백업이 아니다.

재현하려면 기록된 source의 **clean checkout**에서 다음 명령을 사용한다.
기존 결과는 불변이며 새 output 경로가 필요하다. 이는 재현용 예시이고,
실패 후 새로운 파일럿을 추가 실행한 것은 아니다.

```bash
python -m scripts.run_cs_saf_pilot cpu \
  --cache-root artifacts/cs_saf/prepared_v1 \
  --output artifacts/cs_saf/cpu_reproduction
python -m scripts.run_cs_saf_pilot pilot \
  --cache-root artifacts/cs_saf/prepared_v1 \
  --cpu-gate artifacts/cs_saf/cpu_reproduction/COMPLETE.json \
  --output artifacts/cs_saf/pilot_reproduction --gpus 0,1,2,3
```

원 실행 Python은 `/home/finx_sbk/.conda/envs/cofseq/bin/python3`,
PyTorch는 `2.1.2+cu121`이다. GPU 환경에서 동일 source/seed가 다른
하드웨어까지 bitwise 재현된다는 주장은 하지 않는다.
