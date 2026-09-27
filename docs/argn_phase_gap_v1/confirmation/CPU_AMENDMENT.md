# 승인 검토에 따른 CPU 실행 — 2026-09-27

GPU1 추가 확인 dispatch는 자동 승인 검토에서 거절됐다. 사전 허용 GPU2·3
이외의 공유 GPU1 사용이 승인되지 않았다는 이유다. 해당 명령은 실행되지
않았다. 이를 우회하지 않고 CUDA_VISIBLE_DEVICES를 빈 문자열로 고정한
CPU runner에서 같은2부모×2후보×2추가 seed의8개 생성을 수행한다.

원래 GPU1 프로토콜/manifest/스크립트는 미실행 등록 이력으로 보존한다.
새 CPU runner와 이 amendment를 별도 manifest에 고정한다. 데이터, 모델,
GMM, sampling seed, 평가 및 추가 학습0이라는 조건은 그대로다. CPU/GPU의
부동소수점·난수 구현 차이가 출력에 영향을 줄 수 있으므로 **CPU에서 고정
기준 onset_fit도 같은 seed로 다시 생성한다**. 따라서 총12회 추가 생성이다.
기존 GPU 기준과 같은 거래열이라고 가정하지 않는다. CPU 세 팔 사이에서
전체 label/길이 일치를 검증하며, seed별 CPU 결과를 별도로 보고한다.

기존 GPU 결과와 CPU 결과를 하나의 동일 장치 반복으로 합치지 않는다.
CPU 대조의 방향이 같으면 장치·추가 draw를 바꾼 강건성 근거로만 사용한다.
최종 test 접근0. 공유 GPU 신규 점유0. 두 CPU worker를4thread씩 실행한다.
