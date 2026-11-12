# 실행과 비교 재현

작업 checkout은 `research-argn-state-first`, Python은
`../research-reporting/argn-state-first-2026-09-27/runtime/bin/python`이다.
`PYTHONPATH=.:scripts`, BLAS/OMP/MKL thread는 각각4를 사용했다.
모든 기존 protocol/모델/생성 결과는 그대로 남아 있다.

## 입력과 적합

- `artifacts/argn_residual_v1/metadata_optimization.parquet`: GMM/보정 적합.
- 같은 디렉터리의 `metadata_internal_validation.parquet`: 내부 진단만.
- 보정 학습은619고객의 처음 최대1024거래, 첫 gap 제외513464건.
  실제 label 경로를 고정한 시간 시뮬레이터이며 전체 자유 생성으로 세지 않는다.
- 내부 진단은69고객/59814건. 개발147고객의 실제 거래 속성을 적합에 사용하지 않았다.
- 기존 `artifacts/argn_clock_regression_v1/time_head.pt`는 변경하지 않는다.
  location 팔의 weights/means/covariances는 원본 buffer와 정확히 같다.

새로운 source와 protocol의 SHA는 각 artifacts/docs의 MANIFEST.json에 있다.
`run_argn_clock_evolution.py prepare`는4개 진행 조건 GMM과2개 위치 보정을,
`run_argn_clock_location.py prepare`는원래 GMR 위의2개 위치 보정을 적합한다.
이미 적합된 디렉터리에서 prepare를 다시 실행하면 overwrite를 거부한다.
재적합하려면 별도 명시된 새 프로토콜/출력 경로를 사용한다.

각 prepare는 첫 자유 생성 전에 완료됐다. location 추가는 evolution의 내부
조건부 진단을 근거로 등록했으며 evolution 자유 생성 점수를 비교하기 전이다.
이 선택 과정을 미리 정했던5팔로 소급 기술하지 않는다.

## GPU 생성

두 script의 `dispatch`는 유휴 GPU를 두 차례 확인한 후 worker를 시작한다.
evolution의 pending queue가 모두 실행된 뒤 location dispatcher를 시작하여
두 dispatcher가 아직 초기화 중인 같은 GPU를 동시에 선택하지 않게 했다.
기존 작업 GPU1/2가 점유 중일 때 location 첫 두 worker는빈 GPU0/3을 사용했다.

- evolution: 3팔×2부모×4draw =24새 거래열.
- location: 2팔×2부모×4draw =16새 거래열.
- native 부모:20260930,20261001. 이것은 새 시간 head의 독립 재학습 seed가 아니다.
- GPU draw:20261011,20261012,20261021,20261022.

각 `runs/{arm}_{parent}/GENERATION_{draw}.json`에 raw 파일/사용 head SHA,
행 수, 시간, sampling/clipping audit와 전체 labels/lengths의 동일성을 기록한다.
`worker_{parent}/{arm}/ROUTING_CHECK.json`은 실제 native checkpoint에 연결한
첫 gap, 시간 토큰, 배치 순서 변경/축소 후 clock 상태 검증이다.
모든 팔의 첫 gap은 기존 native 출력이고 새 모듈에 미래 길이/실제 clock을 넣지 않는다.

## 평가

`report_argn_clock_evolution.py`, 이어서 `report_argn_clock_location.py`를 실행한다.
normal/fraud gap·amount와 개인 amount는 raw parquet에서독립 구현으로 재계산한다.
개인 gap은 동일 strict-past20/min5 metric으로 계산하며 희귀 onset 표본 수를
`personal_gap.csv`에 함께 남긴다. 첫 단계의 완료 팔은 SHA가 고정된 독립 audit를
재사용하고 raw 파일 해시를 다시 대조한다.

주요 출력은 RESULTS.md, fit_means.csv, screen.csv, all_models.csv,
clock_trajectory_means.csv, clock_location_tradeoff.{png,pdf,svg}이다.
CSV의 arm/fit_seed/generation_seed로 같은 조건의 비교를 직접 재현할 수 있다.
development 반복을 새 독립 test로 부르지 않는다. 최종 test는 계속 미사용이다.

## 범위

진행 조건 추가, GMR, 두 계수 위치 보정, 시뮬레이션 분포 맞춤 자체는 알려진
방법이다. 이 실험에서의 추가 효과/비용으로 구분해야 한다. 48목적 평가의 고정
예산을 소진한 적합은 optimizer convergence와 구분해 fit 기록에 남긴다.
최신 전체 생성기 직접 비교, 추가 독립 native 학습, 다른 데이터셋, 최종 test는
이 시간 출력 비교와 별도이며 완료했다고 표시하지 않는다.
