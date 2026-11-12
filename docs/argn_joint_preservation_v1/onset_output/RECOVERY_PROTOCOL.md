# 추가 seed 검증 경로 수정 — 2026-09-27

최초 추가 검증의 두 worker는 gs20261021의 native/frame parquet 저장 후,
공유 helper가 이전 `argn_gap_episode_v1`에서 같은 seed의 비교 결과를 찾다가
중단했다. 그 폴더에는 초기 draw만 있으며, 신규 draw 기준은
`argn_joint_preservation_v1/confirmation`에 있다. **학습/모델 실패가 아니라
생성 후 비교 파일 경로 오류**다. 원래 script, MANIFEST, 실패 로그 및
DISPATCH_COMPLETE(exit1)를 수정·삭제하지 않는다.

복구는 이미 저장된 첫2개 생성의 hash와 올바른 기준의 길이·라벨열을 확인하고,
아직 만들지 않은 gs20261022만 같은 체크포인트·seed·조건·native 구현으로
생성한다. 직접 generator를 호출해 오래된 helper의 비교 경로 의존을 제거한다.
정상 종료한 baseline과 head, 하이퍼파라미터, 데이터 분할을 바꾸지 않는다.
recovery_dispatch의 별도 manifest/launch/완료 기록으로 실패와 재개를 구분한다.
