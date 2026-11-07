# 재실행과 비교 위치

작업 위치는 research-argn-state-first, Python은
../research-reporting/argn-state-first-2026-09-27/runtime/bin/python 이다.
PYTHONPATH=.:scripts, OPENBLAS_NUM_THREADS=4, OMP_NUM_THREADS=4, MKL_NUM_THREADS=4.

실행 순서는 local_source/run_stratified.py initialize → prepare → local_source/dispatch.py
→ local_source/report_stratified.py다. 초기화와 기존 worker 폴더 덮어쓰기는 막혀 있다.
그대로 재실행하면 새 결과가 이전 결과를 덮지 않고 중단한다. 재현 시 별도 artifact 이름을
사용하고 변경된 경로를 새 manifest에 등록해야 한다.

현재 PIPELINE_START.json, logs/fit.log, logs/gpu.log, logs/report.log에 실제 실행이 남는다.
MANIFEST.json은 source/직전 head/입력 자료/프로토콜을 fit 전에 고정한다.
EVALUATION_MANIFEST.json은 평가 source가 새 개발 생성 전에 고정됐음을 기록한다.
GPU UUID/PID는 LAUNCH.json, 완료 여부는 report/DISPATCH_COMPLETE.json에서 확인한다.

같은 팔의 이전 source/학습/생성은 ../argn_clock_adaptive_v1 아래에 있다.
사용자가 보존하도록 한 기존 개선본은 ../argn_clock_location_v1/clock_prefix/time_head.pt다.
두 기준을 혼동하지 않는다. paired_*는 손실 변경의 효과, screen.csv는 보존 모델을
교체할 만큼 결과가 좋은지를 판정한다. native ARGN 재학습은 없고 보정 계수4회 적합이다.
