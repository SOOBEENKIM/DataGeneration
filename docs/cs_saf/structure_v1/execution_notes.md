# 구조 비교 실행 기록

- 첫 CPU gate는 검사 코드의 2차원 boolean mask와 차원 추가를 한 인덱싱 표현에 넣어 중단됐다. `previous[mask][:,None]`로 검사 인덱싱만 수정했다. 모델/설정은 그대로이며 tiny training 및 과학적 학습 이전이었다. 최초 실패 로그는 `artifacts/cs_saf/structure_v1/technical_logs/cpu_gate_attempt01.log`에 보존한다.
- 두 번째 gate는 tiny U의 validation이 epoch0을 선택하여 best의 train loss 감소가 7%에 그친 데서 중단됐다. 실제 학습 loss는 25epoch 동안 약8.6→4.3으로 감소했으나 작은 validation에는 과적합했다. CPU 학습 동작 검사를 validation-best의 train loss로 판단한 것이 부적절했다. 같은 데이터·시드·25epoch·학습률을 유지하고, 최종 가중치 train NLL의 10% 감소와 validation-best 선택의 정확성을 별도로 검사하도록 고쳤다. 원래 tiny 실행/로그는 `smoke_attempt02_best_train_gate_failed`와 `technical_logs/cpu_gate_attempt02.log`에 보존한다. 과학적 학습 설정/판정 기준은 변경하지 않았으며 아직 과학적 학습은 시작하지 않았다.
