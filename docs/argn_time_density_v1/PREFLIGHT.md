# 실행 전 구현 검증

- `tests/test_argn_time_density.py`: 3 passed. 엄밀한 strict-past clock의 vectorized/
  recurrent 일치(0 gap, 상한, 부족 이력, ring wrap, batch 재정렬·축소), 정수 구간
  확률 총합1, ±30 표준편차 tail의 유한 gradient, 상대 출력의 위치 복원, private RNG.
- 실제 frozen ARGN을 CPU에 strict loading한10step 통합 점검: 첫 gap logits28개
  완전 일치, 새 sampled digit210개 완전 일치, clock 매step 완전 일치.
  7step 후 batch를 [3,0]으로 축소/재정렬한 뒤에도 일치했다.
- GPU 학습 worker도 각 선택 head마다 같은 통합 검증을 다시 수행한다.

이는 데이터 생성 품질 향상 결과가 아니라 실행 전 구현 검사다.
