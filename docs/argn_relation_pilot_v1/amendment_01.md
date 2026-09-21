# 실행 준비 메타데이터 누락 정정

최초 사전등록 `afd418c` 이후 A/G/R이 모두 모델 초기화 전
`KeyError: is_sequential`로 종료됐다. 학습 update와 생성은 각각0회였다.
OriginalData를 복사했지만 공식 codec 통계가 ModelStore/tgt-stats 및 ctx-stats에
저장되어 있어 누락됐다. 모델 구조·예산·seed·판정 기준에 따른 과학적 실패가 아니다.

부호화 원자료와 함께 **인코딩 통계만** 동일하게 복사하고 hash를 검사하도록
수정했다. 과거 모델 가중치/optimizer는 복사하지 않음을 검사한다. 원 실행 기록은
artifacts/argn_relation_pilot_v1/excluded_initialization 아래 보존한다.
CPU workspace 검사1개를 추가하고 모두 통과한 뒤 동일 설정으로 시작한다.
등록 문서의 OriginalData 복사는 이 메타데이터까지 포함하는 것으로 정정한다.
