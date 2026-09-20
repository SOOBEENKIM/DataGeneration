# 구조 비교 실행 기록

- 첫 CPU gate는 검사 코드의 2차원 boolean mask와 차원 추가를 한 인덱싱 표현에 넣어 중단됐다. `previous[mask][:,None]`로 검사 인덱싱만 수정했다. 모델/설정은 그대로이며 tiny training 및 과학적 학습 이전이었다. 최초 실패 로그는 `artifacts/cs_saf/structure_v1/technical_logs/cpu_gate_attempt01.log`에 보존한다.
