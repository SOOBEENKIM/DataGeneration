# 실제 이력에서의 금액 조건부 생성 진단

전환 참조 생성8회 결과를 본 뒤 후속 진단으로 등록한다. 새 최종 검정이 아니다.
이 문서 작성 후 실행. 두 B_event_label_first checkpoint를 그대로 쓴다.

- optimization과 development의 모든 사기 위치, 각각 동수의 정상 위치를
  고정 seed20260927로 추출한다. 같은 선택을 두 checkpoint에 사용한다.
- 실제 전체 과거의 ARGN history와 static context를 사용한다. 현재 길이/position,
  label/category/gap/merchant를 실제값으로 고정한다. amount digits만 모델
  생성 순서대로 sampling한다. 해당 거래 이후의 dummy recurrent state는 버린다.
- 관측 label 그대로/label만 반대로 바꾼 경우를 비교한다. 후자는 모델 민감도이며
  현실의 인과 효과 또는 원래 분포에서 타당한 counterfactual로 해석하지 않는다.
- 각 위치에서4회 sampling, seed20261101–04. 동일 정보에서 teacher amount digit
  NLL도 정상/사기로 집계한다. 최고 draw 선택 없음.
- 금액은 DIGIT 토큰의 직접 수치 복원으로 비교한다. 원형 decoder의 보호된
  최상위 꼬리 재표본추출 전 값이며, 현재/실제 양쪽을 같은 방식으로 복원한다.
  실제 화폐 금액의 완전한 end-to-end 생성 결과라고 부르지 않는다.
- 이 진단은 실제 현재 앞 필드를 주는 충분한 정보 조건이다. 나쁘면 금액 출력
  학습 부족의 근거가 되고, 좋으면 자유 생성 입력/이력의 추가 대조로 이어간다.
  test 이벤트 미접근, 기존 가중치 미변경, 모든 선택·출력 해시 보존.
