# 2026-09-29 실행 복구

이전 pipeline PID1907476와 dispatcher는 현재 프로세스 목록에 없었으며, 마지막 큐 갱신은2026-09-28 15:08:26UTC였다. 당시 pending8/running0, LAUNCH와 generation 결과가 없었다. 종료 원인을 확정할 로그는 없어 추정하지 않는다. 전날 자동 대기 안내는 당시 살아 있던 프로세스만 확인한 것이며 실제 후속 실행 완료를 뜻하지 않았다.

현재 외부 compute 작업 없이 GPU0–3이 모두 idle임을 확인했다. 새 tmux 세션에서 같은 frozen dispatcher/evaluator를 재개했다. 첫 재개 시도는 virtualenv Python 심볼릭 링크를 resolve한 launcher 오류로 mostlyai import에서 종료했다. 이를 모델·데이터 문제와 구별하고 기존 실패 로그를 보존했다. 원래 virtualenv 경로를 그대로 사용해 import를 확인한 뒤 재실행했다. frozen source/protocol/head에 변경은 없다.

두 번째 재개에서는8worker가GPU0–3에 시작됐고 실제GPU 사용률58/52/50/61%를 확인했다. 새 세션 argn-feedback-20260929, pipeline PID2109527. STATUS.json과 PIPELINE_STATUS.json이 최신 상태다. 완료를 확인하기 전에는32생성 또는 최종 평가 완료라고 보고하지 않는다. 모델 교체/commit/push 없음.
