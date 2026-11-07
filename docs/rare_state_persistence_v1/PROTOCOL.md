# rare_state_persistence_v1 — 1단계 프로토콜 (등록 2026-11-07)

연구 질문: 고객별 거래열 생성기는 사기 구간(드문 상태 구간)의 지속·시작·종료
동역학을 보존하는가. 1단계는 **새 학습·새 생성 없이** 기존 development 생성물에
표준 집계 지표와 상태 조건부 지표를 같은 표에 놓고, 실제 대 실제 바닥으로
각 차이를 정규화한다. 최종 test는 읽지 않는다.

## 데이터

- 실제 development: `research-argn-fraud-audit/artifacts/sparkov_argn_control_v2/prepared/validation.parquet`
  (147명, 177,997건, 사기율 .644%). 모든 생성물은 이 147명을 대상으로 생성된 것이다.
- 실제 바닥 풀: 같은 폴더의 `train.parquet` (688명 = optimization 619 + internal 69,
  916,567건). development와 고객이 겹치지 않는다.
- 생성물 등록표는 `scripts/run_rare_state_persistence.py`의 `registry()`에 고정했다.
  가족: `native_argn`(원형 ARGN·각 후보의 native 출력), `native_argn_config`(숫자 인코딩·열 순서만 변경),
  `state_first`(B/B+S/가중치/라벨 우선), `output_heads`(금액·업종 head), `episode_heads`
  (gap/episode/transition head), `time_calibration`(H/P6/C/init_only/clock_prefix/personal_six/mixture),
  `other_generator`(CPAR 2종, LBDTPP 2종, MeLD500). 없는 파일·읽기 실패는 `SKIPPED.csv`에 이유와 함께 남긴다.
- CPAR_128_CPU는 supported 143명 생성물이므로 147명 실제와 고객 집합이 다르다. 고객 수 관련 지표는
  그 행에서 해석하지 않는다. LBDTPP는 금액·가맹점이 없어 해당 지표가 NaN이다.

## 지표 (`benchmarks/rare_state_persistence.py`)

표준(S): 사기율 절대차, 업종 TV, 금액 log-W1, 간격 log-W1, 고객 길이 log-W1, 고객당 가맹점 수 상대차,
실제 10분위 구간의 열 쌍 TV 4개(업종×금액, 간격×금액, 업종×라벨, 라벨×금액).

상태 조건부(R): 사기 지속률(사기 다음 사기 확률) 절대차, 시작률 상대차, 구간 길이 log-W1,
구간 지속 시간 log-W1, 사기 경험 고객 수 상대차, 사기/정상 간격 log-W1, 사기/정상 금액 log-W1,
사기/정상 업종 TV, 사기 시작 개인 대비 간격 log-W1, 구간 나이 구간별 종료 위험 표.
구간 정의는 기존 `benchmarks/argn_state_evaluation.episode_features`(2026-09-27 등록)를 그대로 쓴다.

## 실제 대 실제 바닥

train 688명에서 147명을 비복원 추출(seed 20261107, 200회)해 development 147명과 같은 지표를
계산한다. 각 지표의 중앙값·p95를 바닥으로 두고, 생성물 차이 ÷ 바닥 중앙값을
`DEGRADATION_RATIO.csv`에 기록한다. 바닥은 "같은 크기의 실제 표본 둘이 얼마나 다른가"이며
학습 노출 여부와 무관한 표본 잡음이다.

## 판정 조건 (RESEARCH_DECISIONS.md 2026-11-07 항목 5)

- 표준 지표 비율 ≤ 2 이면서 지속률·구간 길이 비율 > 5 → 가설 (a) ARGN 부분 확정, 2단계로.
- 표준 지표 비율도 > 5 → "표준 지표가 가린다"는 주장 철회, 일반 품질 문제로 분류.
- 바닥 자체가 지속률 차이와 같은 크기 → 147명으로는 판정 불가로 기록.

## 산출물

`ALL_GENERATIONS.csv`, `MEANS.csv`, `HAZARD_BY_AGE_BAND.csv`, `FLOOR_DRAWS.csv`, `FLOOR_SUMMARY.csv`,
`DEGRADATION_RATIO.csv`, `DEGRADATION_RATIO_MEANS.csv`, `SKIPPED.csv`, `MANIFEST.json`(입력 해시, 횟수,
새 학습 0, 새 생성 0, test 미읽음). 해석은 `RESULTS.md`에 실행 후 작성한다.
