# 실행 순서와 환경

기존 결과 디렉터리를 덮어쓰지 않는다. 아래 학습/전처리 명령은 새 checkout과
새 artifact 폴더에서 실행한다. seed는 재현 식별자이며 달력 날짜가 아니다.

## 입력과 환경

`scripts/run_sparkov_argn_control.py`의 `SOURCE`는 이 작업공간의 기존
`data/cof_seqgen_saf/canonical/sparkov`를 가리킨다. 다른 위치에서는 이 경로를
명시적으로 조정한다. canonical 분할은 고객 기준 train 688 / validation 147 /
test 148명이며, 이 실험은 train/validation 고객의 이벤트만 필터링해 읽는다.
`roles.parquet`는 로컬 내부 ID 연결용이며 공개 결과 파일이 아니다.

ARGN은 Berka 재현에서 검증한 mostlyai-engine 1.0.4, torch 2.5.1+cpu,
pandas 2.2 / NumPy 1.26 환경을 사용했다. CPAR은 별도 SDV 1.38.0,
DeepEcho 0.8.1, torch 2.1.2+cu121, pandas 2.3.3 환경에서 CUDA를 비활성화했다.
`ARGN_PY`와 `CPAR_PY`를 해당 환경의 python 절대경로로 설정한다.
수정한 설치 라이브러리를 사용하지 않는다. 실행 중 격리한 함수 변경과
동등성 검사 범위는 [CPU 실행 기록](CPAR_EXECUTION_AMENDMENT.md)에 남겼다.

## 데이터 준비, 기본 대조

```bash
"$ARGN_PY" scripts/run_sparkov_argn_control.py prepare
"$ARGN_PY" scripts/run_sparkov_argn_control.py fit --arm parent --seed 20260928
for seed in 20260928 20260929; do
  "$ARGN_PY" scripts/run_sparkov_argn_control.py fit --arm native --seed "$seed"
  "$ARGN_PY" scripts/run_sparkov_argn_control.py fit --arm relaxed --seed "$seed"
  "$ARGN_PY" scripts/probe_sparkov_argn_control.py "native_$seed"
  "$ARGN_PY" scripts/probe_sparkov_argn_control.py "relaxed_$seed"
  "$ARGN_PY" scripts/probe_sparkov_generation_order.py "$seed"
done
"$ARGN_PY" scripts/run_sparkov_control_comparators.py empirical
"$ARGN_PY" scripts/prepare_sparkov_context_support.py
PYTHONHASHSEED=20260928 CUDA_VISIBLE_DEVICES='' "$CPAR_PY" scripts/run_sparkov_control_comparators.py cpar
```

ARGN `fit`은 학습 뒤 validation/synthetic 두 context 모집단에 대해 각 두
생성 seed를 실행한다. `parent`는 공통 합성 context를 만든다. CPAR은 128 epoch
학습 뒤 native 범주형 context가 지원하는 validation 143명으로 두 번 생성한다.
전체 147명 중 새 지역 값 4명에 대한 native 오류와 제외 기준은
[공통 지원 조건](CPAR_CONTEXT_SUPPORT_PROTOCOL.md)에 기록했다. 실제 미래 길이나
라벨은 생성 조건으로 공급하지 않는다. 재표집 비교의 길이는 학습 고객의
경험분포에서 뽑는다.

열 생성 순서 대조는 engine의 private `_resolve_gen_column_order` 함수를
메모리에서 일시 대체한 진단이다. 기존 flexible-order 모델을 사용하지만
1.0.4의 공개 API에 category-first 플래그가 있다는 뜻은 아니다. 설치 소스와
가중치는 보존하며 호출 후 함수를 복원한다.

최종 코드의 CPAR 생성은 검사한 recurrent-cache 계산과 공통 지원 context를
사용한다. 이 규칙을 추가하기 전에 시작한 원래 학습 프로세스는 완료된
`model.pkl`/`FIT.json`을 보존하고 native 생성 단계에서 종료한 뒤,
`scripts/sample_sparkov_cpar_control.py`로 이어갈 수 있다. 같은 생성물을 동시에
쓰는 프로세스를 실행하지 않는다. 생성 스크립트는 실제 모델의 context encoder로
지원 여부를 재확인하고 완료된 seed를 건너뛴다. 각
`generation_supported_validation_*.json`에 실행 방식을 기록한다.

## 문제 발견 후 추가한 전처리 대조

```bash
"$ARGN_PY" scripts/probe_sparkov_gap_codec.py
"$ARGN_PY" scripts/run_sparkov_argn_control.py prepare-digit
for seed in 20260928 20260929; do
  "$ARGN_PY" scripts/run_sparkov_argn_control.py fit --arm digit_relaxed --seed "$seed"
  "$ARGN_PY" scripts/probe_sparkov_generation_order.py "$seed" --source-arm digit_relaxed --priority category
done
"$ARGN_PY" scripts/prepare_sparkov_numeric_digit.py
for seed in 20260928 20260929; do
  "$ARGN_PY" scripts/run_sparkov_argn_control.py fit --arm numeric_digit_relaxed --seed "$seed"
  "$ARGN_PY" scripts/probe_sparkov_generation_order.py "$seed" --source-arm numeric_digit_relaxed --priority category
done
```

DIGIT은 공식 encoding 설정이다. 기본 모델 크기와 numeric subcolumn 수까지
변하므로 neural fit 비교를 순수한 정밀도 한 요소의 효과라고 부르지 않는다.
같은 원본의 roundtrip 비교가 학습 이전의 표현 손실을 따로 확인한다.
나머지 열의 통계와 context 통계, 내부 619/69 고객 분할은 고정한다.

기본/캡 해제 체크포인트별 검증 window·열 순서 무작위성 대조:

```bash
for seed in 20260928 20260929; do
  for arm in native relaxed; do
    "$ARGN_PY" scripts/probe_sparkov_validation_noise.py "${arm}_${seed}"
  done
done
```

이 결과를 이용한 새 stopping rule의 학습은 이번 범위에 없다.

## 평가와 검증

```bash
"$ARGN_PY" scripts/evaluate_sparkov_argn_control.py reference
"$ARGN_PY" scripts/evaluate_sparkov_argn_control.py evaluate
"$ARGN_PY" scripts/diagnose_sparkov_fraud_runs.py
"$ARGN_PY" scripts/evaluate_sparkov_argn_control.py reference-supported
"$ARGN_PY" scripts/evaluate_sparkov_argn_control.py evaluate-supported
"$ARGN_PY" scripts/diagnose_sparkov_fraud_runs.py --supported
"$ARGN_PY" scripts/verify_sparkov_argn_control.py
"$CPAR_PY" scripts/summarize_sparkov_argn_control.py
"$CPAR_PY" scripts/summarize_sparkov_argn_control.py --supported
"$CPAR_PY" scripts/plot_sparkov_argn_control.py
"$CPAR_PY" scripts/plot_sparkov_argn_control.py --supported
CUDA_VISIBLE_DEVICES='' "$CPAR_PY" -m pytest -q tests/test_cs_saf_cpar_loss.py tests/test_sparkov_cpar_dense.py tests/test_sparkov_cpar_sampling.py tests/test_sparkov_control_metrics.py
```

결과 행마다 사용한 parquet 해시를 `evaluation_manifest.json`에 남긴다.
검증은 데이터 분할, native/캡 해제의 첫 14 epoch 동일성, 고정 가중치,
생성 순서 대조의 고객별 길이 동일성, 전처리 통계 고정, 출력 연결 키와
CPAR 128회 갱신을 확인한다. 생성 품질이 나쁜 행을 제거하는 검증은 아니다.

`supported_` 평가에서는 모든 비교 모델을 같은 143명으로 제한한다. 나머지
평가 파일은 원래 147명 ARGN 대조를 유지한다. runner는 추가 대조가 도입될 때
확장되었으므로 개별 START의 실행 당시 script hash와 최종 통합 runner hash는
다를 수 있다. 고정한 공식 패키지 소스, 변경 diff, 원본/생성 데이터와 가중치
해시, 학습 진행 기록을 함께 제공한다.
