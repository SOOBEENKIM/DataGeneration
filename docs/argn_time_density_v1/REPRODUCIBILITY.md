# 실행과 검증 경로

모든 명령은 research-argn-state-first 작업 디렉터리에서 지정 runtime Python으로
실행했다. `PYTHONPATH=.:scripts`, BLAS/OMP/MKL thread4를 사용했다. 모델과
생성 원본은 artifacts에 보존하고, Git에는 코드·프로토콜·집계·hash·그림만 저장한다.

## 등록된 주 실험

`scripts/run_argn_time_density.py prepare`가 학습 전에 코드/자료를 동결한다.
`dispatch`는2회 idle 확인 후GPU0/1에서 부모별 worker를 실행했다. 각 worker는
실제-prefix cache replay→3head 학습→조건부 진단→실제 native 통합 검사→12개
자유 생성→기존 평가 순으로 실행했다. 완료 artifact를 덮어쓰지 않도록 재실행은
차단한다. 반복 연구는 새 실험 디렉터리/프로토콜로 등록해야 한다.

`scripts/report_argn_time_density.py --controls-only`는 기존3비교군의4draw를
검증한다. `scripts/report_argn_time_density.py`는 새3팔을 포함해 독립 raw 지표와
사전 비용 screen을 계산한다. 먼저 끝난 arm의 검증 cache를 사용해도 매 raw 파일
hash를 다시 확인한다. GPU 결과와 이전 CPU 확인 결과는 혼합하지 않는다.

## 추가 알려진 GMR 대조

`scripts/run_argn_clock_regression.py prepare`가 별도 protocol/manifest로 적합한다.
fixture precision 수정은 원래 파일/manifest를 보존한 AMENDMENT_01로 기록했다.
실제 GPU2/3 dispatch는 **`scripts/run_argn_clock_regression_amended.py dispatch`**다.
원래 launcher 대신 이것을 사용해야 수정 fixture hash까지 검증한다. 모델은 동일하다.

`scripts/report_argn_clock_regression.py`는 GMR8생성을 독립 검증하고 주 실험의
48개 비교 결과와 합쳐56개 matched-GPU 결과를 보고한다. 이것은 독립 native
ARGN56학습이 아니다. 고정 부모2개, 새 작은 신경망6적합, GMR 전환별4적합이다.

## 조건부 오류 분리

`scripts/intervene_argn_clock.py`는 고정 실제25prefix/500label에서 실제 clock과
생성 clock만 바꾸는 진단이다. 일반 free-generation 결과로 계산하지 않는다.
`scripts/diagnose_argn_time_density.py`는 학습0step 선택과 장기 clock 변화를
기술한다. GMR report가 추가 모델을 같은 진단에 포함한다.

최종 test 이벤트를 읽지 않았다. learning curve와 실패한 설정/품질 비용을 모두
보존했다. 본 회차는 Sparkov 개발실험이며 다른 데이터셋·최신 생성기 전체 직접
비교나 신규성 검증을 대신하지 않는다.
