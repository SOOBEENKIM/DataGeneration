# 검증 기록

- `python -m pytest -q tests/test_argn_transition_reference.py tests/test_argn_label_first_control.py tests/test_argn_label_first_dispatch.py tests/test_argn_state_first.py tests/test_argn_event_weight_control.py`: **13 passed**.
- 추가/진단 runner와 transition reference adapter의 Python syntax 확인 통과.
- 단순 head 확률1,094,506위치의 독립 표 집계 일치, 동일 teacher 입력에서 label
  외 logits bitwise 일치, parent checkpoint key 동일.
- 새 자유 생성8개·1,579,872행의 primary5지표를 별도 loop로 계산해1e-12 이내 일치.
- 조건부 금액 진단의 첫 digit teacher/generation 일치23,540위치; 두 모델의
  위치 선택 파일 hash 동일. sampling4회×2 label조건×11,770위치×2모델.
- 원래 모델10개와 후속 진단 parent2개 가중치 hash 유지. final test 거래 미접근.
- GPU2/3은 각 시작 전 두 번 idle 확인 후 사용했고, 마지막 진단 완료 후
  프로세스 종료와 GPU memory52/93MiB·utilization0%를 확인했다. GPU0/1은 외부 작업.

실행 환경: 기존 고정 `argn-state-first-2026-09-27/runtime/bin/python`,
torch2.5.1+cu121, mostlyai-engine1.0.4. 이번 새 신경망 학습0회.
