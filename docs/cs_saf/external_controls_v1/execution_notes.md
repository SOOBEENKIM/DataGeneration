# 실행·검산 기록

출발점은 GitHub와 작업공간이 일치한 **34ee431bdf55777cbe36c2422d736bc1bf96123c**다. main이나 과거 결과를 덮어쓰지 않았다. 등록/과학 소스 **e1d44f2a6028972e30f1e3f9f7ce0c6dd0b5dd3d**를 2026-09-21 20:28:37 KST에 커밋하고 같은 브랜치에 push했다. 이후 등록한 14개 신경 모델을 학습했다.

## 실제 실행

- U/G × 금액만·행동만·둘 다 × 2자료 = 12fit, 일반 D × 2자료 = 2fit. 모두 새 학습이다. 과거 random-initial tensor를 공통 부분의 동일 초기화에 사용했으며 이전 학습 checkpoint에서 이어 학습하지 않았다.
- 각 모델에 fit-only 6계수 반복 보정을 1회 적용했다. 총 14개 보정 fit이다. raw가 주 분석이며 보정은 보조 분석이다.
- 모델마다 두 variant × 생성 seed 2개, 총 56개 생성물과 5,525,464거래. 자료별 같은 128개 context와 원래 길이 계획을 끝까지 사용했다.
- 과학 실행의 실패/중단/재시도는 0이다. 설정·임계값·시드·성분 수를 실행 중 바꾸지 않았다. 모든 START.json이 과학 파일의 SHA256을 보관하며 checkpoint 감사에서 최종 파일과 일치함을 확인했다.
- Berka U_amount/G_amount/U_both/G_both는 30epoch 상한, 나머지는 내부 check patience로 종료했다. 모든 실행은 20분 시간 상한보다 먼저 끝났다. 같은 epoch 상한이 모든 모델의 최적화 완료를 보장하지 않는다.

## 장치

기존과 같은 finx-System-Product-Name의 RTX 3090 4개를 사용했다. 제한 환경의 장치 미노출과 실제 점유를 구분해서 확인했다. 각 작업 시작 시 사용률<5%·메모리<512MiB 조건을 다시 검사했다. NVIDIA MPS 서비스는 유지했고 다른 작업을 종료하지 않았다.

- GPU 0: Berka U_amount → G_amount → U_both → D_both.
- GPU 1: Sparkov U_amount → G_amount → U_both → D_both.
- GPU 2: Berka U_action → G_action → G_both.
- GPU 3: Sparkov U_action → G_action → G_both.
- 학습과 생성 종료 후 빈 GPU 3에서 checkpoint를 읽기 전용으로 감사했다. 독립 생성물 집계는 CPU 1스레드다. 추가 학습·생성은 없다.

[모든 admission과 작업 PID](execution_resources.json), [완료 목록](dispatch_done.json), [실제 예산](training_budgets.csv)을 남긴다. 로그 원본·가중치·생성물은 Git에 올리지 않는 artifacts 경로에 있다.

## 검증

1. CPU 34개 테스트 통과. SciPy 밀도/Jacobian 대조, 0 질량·양수 support, 기존 모델 forward/긴 sampling 일치, strict-past와 32거래 이후 replay, 행동 정규화·보정 범위·gap 경로를 확인했다. 전체 1,990,071개 개발 금액의 유한 loss와 14개 자료/모델 조합의 경계 입력을 확인했다. 실제 0 target은 Berka fit 8/check 1/validation 3, Sparkov 0이며 codec의 정확한 0 원자 판정과 일치했다.
2. GPU 8개 조합의 forward/backward 유한성을 확인했다. optimizer update는 0이며 과학 학습 횟수에 포함하지 않는다.
3. evaluator를 import하지 않는 NumPy histogram/ECDF로 1,064개 스칼라를 재계산했다. 최대 차이 1.4156e-15. 파일 해시·개체·길이·순서·timestamp와 기존 모델/입력/evaluator 소스 6개의 불변성도 확인했다.
4. checkpoint 14개에서 공통 초기값, 최저 check epoch, 실제 update 수, source/file 해시를 확인했다. forward는 재사용하되 NLL·Brier·MAE 집계는 별도로 작성했고 혼합 금액 밀도는 NumPy/SciPy로 계산했다. 28개 raw/보정 평가의 최대 차이는 6.0942e-8이다.

검증 통과는 가능한 모든 오류의 부재나 모델 우수성의 증명이 아니다. 그림은 실제 CSV에서 생성했고 생성 난수 2개 범위를 신뢰구간으로 표시하지 않았다.

## 결과 후 분석과 종료

보고서·독립 검산 스크립트는 결과 처리용이며 과학 runner가 import하지 않는다. 독립 검산에 완료 대기 기능을 추가했지만 모델·평가식·판단 기준은 바꾸지 않았다. 네 조합의 차이와 금액 상위 분위/최댓값은 기술적 분석으로 추가했고 등록된 판정을 대체하지 않았다. Berka의 남은 상단 꼬리를 숨기기 위해 clipping하거나 생성 난수를 재선택하지 않았다.

등록한 비교·검산·보고서 작성을 마친 뒤 종료한다. 새로운 후보나 재현 시드 확대를 자동 시작하지 않는다. 현재 결과로 설명할 수 있는 구조와 기여 주장의 범위를 정리하는 단계다.
