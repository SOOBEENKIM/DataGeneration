# 실행 검증

- 19 targeted tests passed: episode control, amount control, transition reference,
  state input, boundary-aware state evaluation. 기존 테스트도 함께 실행했다.
- 실제 두 부모 checkpoint에서 각 optimization816,283 / internal-validation100,226 /
  development177,997 위치의 7개 gap digit logits를 cache/복제 head와 대조했다.
  총15,323,084 token-position replay 검증. 학습·간격 진단에서는 각 고객 첫 행 제외.
- 실제 checkpoint CPU 통합검증30개 label확률: 온라인 memory에서 생성한 값과
  독립적인 파이썬 label 이력 계산을 대조했다. 고객 batch를 줄이고 순서를 바꾼
  뒤에도 ever-fraud/initial-label이 해당 고객을 따라간다. 최대확률 오차8.35e-8.
  검증용 가상 길이는 이 unit integration에만 썼고 실험 거래열에는 공급하지 않았다.
- 기존 category/amount head tensor를 새 모든 조건에서 elementwise 동일성 검사한다.
  원래 model weight/source/input hashes는 manifest 검증으로 고정한다.
- 결과 보고기는 raw parquet에서 정상/사기 amount/gap W1을 독립 SciPy 계산해
  기존 평가기와1e-12 이내 일치 검사한다. 생성 길이와 고객이 매칭 기준과 동일하고,
  gap-only label순서는 기준과 같으며 joint label순서는 transition-only와 같다.
  완료 여부/검증 대상은 completion_evidence.json와 independent_checks.csv를 따른다.
- label-only Monte Carlo768draw는 고객 상태 sampling 진단이다. 추가 신경망 학습이나
  전체 거래 생성768회라고 세지 않는다. 파라미터 선택에 사용하지 않았다.
- final-test event attributes unopened. development 결과는 개발/진단 근거다.

실행 명령(저장소 root, 기존 mostlyai-engine1.0.4/torch2.5.1 runtime):

```sh
python -m pytest -q tests/test_argn_episode_control.py tests/test_argn_amount_control.py tests/test_argn_transition_reference.py tests/test_argn_state_first.py tests/test_argn_state_evaluation.py
python scripts/run_argn_gap_episode.py prepare
python scripts/dispatch_argn_gap_episode.py
python scripts/verify_argn_episode_runtime.py
python scripts/probe_argn_episode_rollouts.py
python scripts/audit_argn_episode_length_support.py
python scripts/report_argn_gap_episode.py
python scripts/plot_argn_gap_episode.py
```

prepare와worker는 기존 manifest/run이 있으면 덮어쓰지 않는다. 재실행하려면
원래 결과를 보존한 새 실험 ID를 사용한다. GPU dispatcher는 등록된 GPU2·3만
사용하며 launch직전 실제 다른 연산 process/메모리/사용률을 확인한다.
