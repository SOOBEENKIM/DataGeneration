# ibm_sealed_test_v1 — IBM 봉인 test 1회 평가 (등록 2026-11-10, test를 열기 전)

사용자 지시(2026-11-10): "봉인 테스트 열기"(Sparkov 결과 보고 후 명시적 확인). `artifacts/ibm_tabformer_v1/prepared/test.parquet`·`test_context.parquet`는 2026-11-08에 기록된 뒤 한 번도 읽지 않았다
(239명). 이 문서를 커밋한 뒤 처음 읽는다. **한 번만** 평가하며, 결과를 본 뒤에는 IBM 모델·표·초매개변수·평가 설정을 바꾸지 않는다. 결과가 나쁘더라도 그대로 기록한다.

## 1. 고정된 것 (모두 dev 단계에서 확정; test로 다시 추정·선택하는 것 없음)

- backbone fit: ARGN `native_20261108/09`, RTF fit 20261108/09(train 1,000명). 생성 seed 20261011/12.
- 표·출력층(모두 train): 전이표, `routed_subst` 사기 출력층, 가맹점 부품 표(게이트·풀), 간격 표, 가맹점 속성 표, 채널 상태 표, 위치 구간 업종 표, 방문율 커널(κ = 8, λ = 1), 희귀 가맹점 표기 규칙.
- 평가 팔(사다리 + 비교군), 각 fit 2 × 생성 2 = 4 생성, **test 고객의 문맥**(가입 연도 등 4개 정적 변수)으로 생성:

| 팔 | 구성 | dev 효용(등록 탐지기) |
|---|---|---:|
| `argn_native` | 원형 ARGN | .003 |
| `argn_minimal` | + 전이표 + 라벨 라우팅 출력층 | .142 |
| `argn_merchant` | + 이력 참조 가맹점(`full`) | .306 |
| `argn_gap` | + 구간 위치 간격 표 | .331 |
| **`argn_final`** | + 가맹점 속성 일관성 + 방문율 보정(채택 모델) | **.491** |
| `rtf_native` | 원형 REaLTabFormer | .290 |
| `rtf_table` | + 전이표(시험한 것 중 가장 높던 비교군) | .448 |

  RTF 두 팔은 dev에서와 같이 **실제 창 길이**를 받는다(test 사건의 고객별 행 수; RTF 쪽의 조건부 이점은 dev와 동일). ARGN은 길이를 스스로 생성한다.
- 평가(dev와 같은 스크립트, 평가 표만 test로): TSTR 효용(`stable` 등록 탐지기; 보조 `no_merchant`·`no_channel`·`logistic`), 1단계 지표, 사기 구조 지표. 바닥·상한은 train에서 239명 부분표본을 뽑아
  test에 대해 다시 계산(상한 20회, 1단계 바닥 200회, 구조 바닥 200회 — dev와 같은 seed 규칙). 탐지기의 업종 어휘는 dev에서와 같이 평가 표(test)의 빈도 상위 40개로 정한다(라벨 미사용).

## 2. 사전 판단 (수치 고정)

- **S1 (효용 재현).** `argn_final`의 test 효용이 test 상한의 **70% 이상**(dev: .491 / .610 = 80%)이면 "재현", 55–70%면 "부분 재현", 55% 미만이면 "미재현(dev 과적합 의심)".
- **S2 (사다리).** test에서 `argn_native` < `argn_minimal` < `argn_merchant` ≤ `argn_final`이고 `argn_final` − `argn_gap` ≥ **+.08**(dev +.16의 절반)이면 "부품별 기여 재현". `argn_gap` − `argn_merchant`는 dev에서
  중립(+.02)이었으므로 부호를 요구하지 않는다.
- **S3 (비교군).** `argn_final` − `rtf_table` ≥ −.03이면 "가장 높던 비교군과 같거나 높은 수준"이 test에서도 성립. fit이 2개이므로 **우월 주장은 어느 결과에서도 하지 않는다.** 생성 간 표준편차도 함께 보고한다.
- **S4 (충실도).** `argn_final`의 표준 지표 최댓값 ≤ 3배(test 바닥 기준), 사기 새 가맹점·구간 내 간격·고객당 가맹점 수 ≤ 3배, 정상 행 속성 일관성 ≥ .99.
- 모든 팔·모든 지표를 그대로 보고한다. S1–S4 중 미달은 미달로 적는다. test를 본 뒤의 추가 실험은 IBM 모델에 대해 하지 않는다(Sparkov 등 다른 데이터 작업은 별개).

## 3. 한계(미리 적음)

- dev에서 부품과 κ를 골랐으므로 dev 수치는 낙관적일 수 있다 — 그것을 재는 것이 이 평가다. test 239명·fit 2개로는 작은 차이를 가를 수 없다.
- test 고객의 가맹점 중 train에 없는 것은 어느 생성기도 내지 못한다(dev에서 정상 행의 15.6%).
- 탐지기 어휘를 평가 표에서 정하는 것, RTF가 실제 길이를 받는 것은 dev와 같은 조건이며 그대로 둔다.

## 4. 실행

`scripts/run_ibm_sealed_test.py`(ARGN 5팔 생성), `scripts/run_realtabformer_ibm.py prepare_test / generate --split test`(RTF 2팔), 평가는 세 스크립트의 `--dataset ibm_test`.
산출 `artifacts/ibm_sealed_test_v1/`, `artifacts/ibm_realtabformer_v1/runs/*/generated_test_*.parquet`, 기록 `docs/ibm_sealed_test_v1/`. GPU가 다른 사용자의 작업으로 점유된 동안에는 기다린다(설정을 바꿔 끼워 넣지 않는다).
