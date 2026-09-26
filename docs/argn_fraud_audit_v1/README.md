# 사기 라벨을 포함한 ARGN 기준 모델과 관계 오류 감사

**2026-09-27 우선순위 정정:** 아래 확장 실험에 앞서 논문과 같은 조건의 재현이 필요했다. 다음 작업은 [저자 공개 Berka 자료·생성물을 이용한 논문 재현 확인](PAPER_REPRODUCTION_CORRECTION.md)이다. 아래 제안된 추가 Sparkov 학습·선택 대조보다 이 확인을 먼저 한다.

2026-09-27. **기존 ARGN 작업을 보존하고, 라벨 포함 공식 기준 모델의 학습2회·생성4회(680,834거래)를 완료했다. 현재 관찰은 ‘평균은 잘 맞고 희귀한 이력만 어긋난다’는 가정을 지지하지 않는다. 기본 행 관계와 전체 사기 비율부터 충분히 재현되지 않았다.** 새 구조의 기여를 주장할 단계는 아니다.

출발점은 GitHub `research/argn-relation-pilot-v1`와 일치하는 `c1463e5`다. 기존 공식 구현 기반·대조 실험·실패 기록은 재사용할 가치가 있어 삭제하지 않았다. 이전 R 경로의 실패 판정도 유지한다. 이번 작업은 `research/argn-fraud-audit-v1`에 분리했다.

## 사용한 기준과 조건

[TabularARGN 논문 v2](https://arxiv.org/html/2501.12012v2)의 순차·정적 고객 조건 구조를 기준으로 [공식 engine](https://github.com/mostly-ai/mostlyai-engine) 설치 버전2.4.0을 사용했다. 논문의 원래 사기 실험을 재현한 것은 아니다. 새로운 이력 경로나 손실을 추가하지 않았다. 이 설치 버전을 2026년 최신 버전이라고 주장하지 않는다.

Sparkov 고객별 `gap → merchant → amount → category → event_is_fraud`를 생성한다. 정적 조건은 성별·지역·출생연도·도시 인구이며 ID는 연결 키다. 미래 전체기간의 사기 여부와 실제 거래열 길이는 주지 않는다. 정상/사기 비율을 강제하거나 생성 라벨을 사후 수정하지 않는다.

fit550명/715,729거래, check138명/200,838거래, validation147명/177,997거래다. codec은 fit만으로 추정했다. Medium, window100, flexible generation, native value protection·최적화·조기 종료를 사용했다. CPU 학습은 각각9.26분/8.55분, 마지막 epoch35/32, 선택 epoch30/27이었다. 최대100epoch/30분 상한에 닿지는 않았다. **조기 종료가 충분한 관계 학습을 보장하지는 않는다.** [사전 실행 기준](PROTOCOL.md), [실제 학습량과 전체 결과](result_tables.md).

## 직접 확인한 결과

| 항목 | 실제 validation | 새 ARGN 네 생성 결과 |
| --- | ---: | ---: |
| 전체 사기 비율 | 0.6438% | 0.3466%, 0.3647%, 1.3963%, 0.9398% |
| 첫 거래의 fit에 없던 merchant–category 조합 비율 | 0% | 80.95~89.12% |
| merchant–category 공동 TV | — | 0.8302~0.8672 |
| 사기 거래의 category–amount TV | — | 0.4496~0.6415 |
| 사기 거래의 gap–이전/현재 category TV | — | 0.4488~0.5833 |
| 무효 라벨·금액·거래간격 비율 | — | 모두0 |

TV는 낮을수록 좋다. fit에 없는 조합을 현실에서 불가능한 거래라고 단정하지 않는다. 다만 원본 validation의 모든 merchant–category 조합은 fit에 있었고, 원본을 native encode→decode한 결과도 이 조합과 라벨을 그대로 보존했다. 과거 A도 첫 생성 거래에서91.4~93.8%의 조합이 fit에 없었다. 따라서 **오류를 긴 생성 과정의 누적만으로 설명할 수 없다.** [과거 위치별 감사](existing_position_curves.csv), [새 위치별 결과](position_curves.csv), [codec 검사](new_codec_roundtrip.json).

사기 관계 TV는 작은 표본의 영향을 받는다. 실제 고객을 반으로 나눈12회 비교에서도 사기 거래의 gap–이전/현재 category TV는0.409~0.464였다. 두 번째 학습의 해당 지표0.449~0.468을 곧바로 구조적 실패로 단정할 근거는 부족하다. 이 참고 비교는 표본 크기가 달라 유의성 검정을 대신하지 않는다. 반면 merchant–category TV의 real–real 참고 범위는0.051~0.058이었다.

5초 이내(0초 초과) 거래는 원본45건 중 사기1건이고 생성에서는25~42건 중 사기0~1건이었다. 이 조건의 비율 한 개를 성공 기준으로 삼을 수 없다. [모든 조건의 빈도·사기 수·사기율](conditional_risk.csv), [5초 구간](positive_gap_at_most_5s.csv)을 함께 보관했다.

## 어느 정보가 쓰이고 있는가

고객147명의 최초512개까지69,086거래에서 실제 과거와 현재 행의 선행 필드를 주고 거래 유형을 예측했다. 생성 탐지기의 성능을 측정한 것이 아니다.

| 고정 체크포인트 진단 | 학습 seed20260927 | 학습 seed20260928 |
| --- | ---: | ---: |
| 원래 실제 이력의 category 정확도 | 32.95% | 39.10% |
| 현재 merchant embedding만 교란 | 13.72% | 13.17% |
| 첫 위치 이후 history 출력만0으로 제거 | 27.87% | 33.85% |
| fit-only merchant별 최빈 category 조회, 동일 표본 | 99.22% | 99.22% |

모델은 merchant와 이력을 사용하지만 기본 관계를 충분히 학습하지 못한 상태다. 조회표는 학습 자료로만 만든 탐색적 비교이며 완전한 거래열 생성기나 새 제안 모델이 아니다. 정보 제거는 분포 밖의 입력을 만들 수 있어 그 차이를 실세계 인과효과나 최적의 짧은 이력 모델 성능으로 해석하지 않는다. [진단 설계](PROBE_PROTOCOL.md), [seed1 전체 예측](probe_seed_20260927.csv), [seed2 전체 예측](probe_seed_20260928.csv).

## 검증 과정에서도 구분해야 할 원인

두 체크포인트를 고정하고 내부 check를48회 재평가했다. 공식 검증은 구간과 열 순서를 다시 뽑으므로 **가중치가 같아도** check loss가 seed1에서1.0477~1.0985, seed2에서1.0367~1.0874로 바뀌었다. 구간과 열 순서를 모두 고정하면 반복 결과가 같았다. 구간만 또는 열 순서만의 변동도 각각 남았다. [2×2 설계](VALIDATION_PROBE_PROTOCOL.md), [전체 변동 요약](validation_noise_summary.csv).

이는 최소 validation loss가 항상 더 좋은 관계 학습 상태를 뜻하지 않을 수 있다는 구체적인 점검 근거다. 다만 이 실험은 학습하지 않았으므로 **‘불안정한 검증이 조기 종료와 관계 오류의 원인이다’까지 입증하지 않았다.** 다음 통제는 같은 구조에서 검증 표본·순서를 고정한 선택과 추가 학습 예산을 분리해 비교하는 것이다. 조건별 평균 loss는 평가하는 구간/순서가 달라 품질 개선량으로 해석하지 않는다.

## 요청한 네 단계의 현재 상태

1. **오류 위치 확인:** 첫 거래부터 큰 행 관계 오류가 있고, 전체 사기 비율도 학습 seed에 따라 달라진다. 희귀 이력에만 한정된 문제라는 가정을 보류한다.
2. **원인 구분:** codec만의 손실과 누적오차만의 설명을 배제할 근거를 얻었다. 모델이 merchant/이력을 전혀 쓰지 않는다는 설명도 맞지 않는다. 학습량, 선택 규칙, 학습 목적과 표현 중 무엇이 주원인인지는 아직 확정하지 않았다.
3. **원인을 겨냥한 생성 방법:** 아직 확정하지 않았다. 우선 동일 구조의 학습 충분성 및 검증 선택 규칙을 대조해야 한다. 단순 학습 설정으로 기본 관계가 복구되면 그 현상을 새 아키텍처의 필요성으로 삼지 않는다.
4. **baseline·제거 대조:** 이번은 공식 ARGN의 라벨 포함 기준 실행이다. 새로운 제안 모델, 그 제거 대조, 라벨 포함 CPAR/IFT-GAN/TabDiT와의 동일 조건 비교는 아직 수행하지 않았다.

이후 대표 baseline은 이번 native 실행을 보존하면서 충분한 학습을 확인한 ARGN 대조를 추가하는 방식으로 잡는다. 라벨을 넣은 것, 학습량을 늘린 것, 이미 알려진 관계를 보정한 것만으로 독창성을 주장하지 않는다.

## 검증과 재실행

[검산 결과](verification.json): 네 생성물의 사기 수·비율, merchant–category TV와 첫 거래 조합 비율을 별도 Counter/집합 계산으로 확인했다. 이력의 미래 누출·고객 경계·라벨 해석·native nullable dtype 테스트4개를 통과했다. 공식 소스, 기존 생성물6개와 새 체크포인트 hash를 확인했다. 무효 값은 삭제하지 않았다.

기본 명령은 저장소 루트에서 아래 순서다. Python은 `mostlyai-engine==2.4.0`, pandas2.3.3, numpy2.2.6, torch2.9.1 환경을 사용한다. 이 worktree에서는 이웃 `research-cs-saf-external-audit/external/cs_saf_external_audit/runtime/bin/python`을 사용했다.

```bash
python scripts/run_argn_fraud_audit.py audit-existing
python scripts/run_argn_fraud_audit.py prepare
python scripts/run_argn_fraud_audit.py fit --seed 20260927
python scripts/run_argn_fraud_audit.py fit --seed 20260928
python scripts/run_argn_fraud_audit.py generate --seed 20260927
python scripts/run_argn_fraud_audit.py generate --seed 20260928
python scripts/run_argn_fraud_audit.py evaluate
python scripts/probe_argn_fraud_training.py old_A
python scripts/probe_argn_fraud_training.py seed_20260927
python scripts/probe_argn_fraud_training.py seed_20260928
python scripts/probe_argn_fraud_training.py lookup
python scripts/probe_argn_validation_noise.py
python scripts/verify_argn_fraud_audit.py
python scripts/summarize_argn_fraud_audit.py
```

`prepare`와 `fit`은 기존 폴더를 덮어쓰지 않고 중단한다. 재학습에는 새로운 출력 위치를 명시적으로 준비해야 한다. `generate`는 완료된 생성물을 재사용하며 불완전한 저장물에서는 중단한다. 원본 데이터와 역할표는 이웃의 기존 실험 경로에 의존한다. 새 머신에서 GitHub clone만으로 실행되는 배포 패키지는 아니다.

코드·설정·집계만 Git에 보관하며 큰 원본/생성 parquet와 가중치는 로컬 `artifacts/argn_fraud_audit_v1/`에 남긴다. [실행·해석 기록](execution_notes.md). 최종 test outcomes는 읽지 않았고, validation은 반복 사용한 개발 자료다. 이번 결과는 최종 독립 검증이나 논문 방법 전체의 우열 판정이 아니다.
