# 우선순위 정정: Berka 논문 재현을 먼저 확인

2026-09-27 사용자 지적에 따른 정정. **공식 engine으로 사용자 데이터에 실행한 것과 논문 benchmark를 재현한 것은 다르다.** 앞선 Sparkov 감사는 원문 재현을 완료하기 전에 시행한 확장 실험이다. 그 결과를 근거로 ARGN 방법의 한계나 새로운 모델의 필요성을 확정하지 않는다. 검증 선택 규칙을 바꾸는 추가 Sparkov 실험보다 아래 재현을 우선한다.

## 원문에서 다시 확인한 사실

[TabularARGN v2, 부록F 표2·3](https://arxiv.org/html/2501.12012v2#A6): flat 자료는 Adult, ACS-Income, Default, Shoppers이고 sequential 자료는 Baseball, California, Berka다. Sparkov는 포함되지 않는다. Default의 채무불이행 라벨은 거래 사기 라벨과 구별해야 한다.

Berka는 account와 transaction 테이블을 사용한다. 외부 train/holdout 계좌는2,250/2,250개, 거래는526,442/529,878행이다. [부록G 표7](https://arxiv.org/html/2501.12012v2#A7)의 비-DP ARGN 점수는 overall accuracy .79, univariate .87, bivariate .68, coherence .82다. 이는 생성 분포 점수이며 사기 탐지 정확도나 이번 teacher category 정확도와 같은 지표가 아니다.

## 저자 저장소와 확보 자료

[mostly-ai/paper-tabular-argn](https://github.com/mostly-ai/paper-tabular-argn), 확인한 commit `97781f9cab4ef91e347dfc31650e38ef8dcf2e56` (2025-02-28).

- `data_train/berka_account_trn.csv`: 실제 읽은 행2,250개, 계좌2,250개. SHA256 `1a920bd05c52c70b1a741a21c1d182c14783a475dbe5111c33a3144533bd151d`.
- `data_train/berka_trans_trn.csv.gz`: 실제 읽은 행526,442개, 계좌2,250개. SHA256 `4571f4ac248ee871b32ef3214ca73554346cbb9dad2206b1624b1966e35aea2c`.
- 저자 공개 비-DP Berka 생성물은5회분의 경로가 존재한다. 먼저 순서상 run1과 합성 account를 평가 검증용으로 확보한다. 이 자료는 우리의 재학습 결과로 세지 않는다.
- 공개 ARGN 순차 실행 코드는 Baseball/California용이다. 확인한 전체 tree에는 Berka 전용 ARGN 스크립트와 dependency lock/requirements 파일이 없다. 따라서 공개 소스가 확인하지 못해주는 전처리·버전을 임의로 ‘논문과 동일’하다고 채우지 않는다.
- 공개 순차 스크립트는 먼저 정적 부모 테이블 모델을 학습·생성하고, 그 합성 부모에 조건화해 자식 거래열을 생성한다. `train(max_training_time=300)` 이외 대부분은 당시 기본값에 의존한다. 우리 실험의 실제 validation 고객 조건, 명시적인 window/batch, 2.4.0 설치 버전과 동일함이 입증된 것은 아니다.
- 공개 flat 평가 notebook은 `mostlyai.qa.report`를 사용한다. 현재 engine 실험 venv에는 `mostlyai-qa`가 설치되지 않았다. 평가 환경은 기존 학습 환경을 변경하지 않고 별도로 구성해야 한다.

자료의 로컬 보관 위치는 worktree의 이웃 `research-reporting/argn-paper-reproduction-2026-09-27/paper-tabular-argn`이다. 표에 기록된 원문 규모와 실제 다운로드 자료가 일치함까지 확인했다. **논문의 학습 및 점수 재현은 아직 완료하지 않았다.**

## 바로잡은 실행 순서

1. 저자 공개 생성물을 논문 지표로 평가해 데이터 읽기·전처리·지표 구현부터 확인한다. 정확한 당시 QA 버전/순차 평가 설정의 미확인 부분을 기록한다.
2. 같은 공개 Berka 분할, 원래 필드와 부모→자식 생성 절차로 기준 모델을 실행한다. 당시 engine 버전·설정이 확인되지 않으면 그 차이를 기록하며 동일 버전 재현이라고 부르지 않는다.
3. 논문 점수와 비교하고 차이가 있으면 실행·버전·데이터 처리·학습부터 대조한다. 그 전에는 Sparkov에 새 학습 규칙이나 모델 구조를 추가하지 않는다.
4. 기준이 확인된 뒤 Sparkov 적용, 라벨 추가, 연구용 세부 관계 평가를 구분해 확장한다.

기존 Sparkov 생성물과 관찰은 보존한다. 오류를 관찰했다는 사실은 유효하지만, 그것만으로 논문 재현 실패 또는 ARGN의 구조적 한계를 입증한 것은 아니다.
