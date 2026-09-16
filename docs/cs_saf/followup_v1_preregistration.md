# 규제 비용·집단 비율·외부 비교: 사전등록 v1

2026-09-16. 사용자 요청에 따른 별도 탐색 확장이다. 기존 replication의 실패와
90회 중단 결정은 그대로 보존한다. 이 문서는 구현·새 결과 확인 전에 고정한다.
[고정 계약](../../configs/benchmark_v2/cs_saf_followup_v1.yaml).

## 범위와 판정

- pi=.05: 기존 U/E/ER 30회를 재사용하고 lambda=.003/.03의 20회만 추가.
- pi=.10/.25/.50: U와 E(lambda=0), .003/.01/.03 전부를 같은 5개 seed·두 kappa에서
  학습한다(150회). 좋은 lambda만 골라 확장하지 않고 전체 곡선을 보존한다.
- 모델 수식·파라미터 수·초기화·AdamW·50 epoch/patience5·global validation NLL
  checkpoint 선택·epoch9 감사는 모두 부모 계약 그대로. 각 모델 fresh fit.
- 기존 response/accuracy 기준은 보조 진단으로 그대로 계산한다. 통과 여부에 관계없이
  모든 등록 cell을 완료한다. 기존 실패를 성공으로 재분류하거나 새로운 허용 오차를 만들지 않는다.
- 모든 lambda/시드/집단 비율을 공개한다. 5개 paired seed mean/SD/SE/95% t interval은
  동일 데이터에서 학습 난수에 대한 기술 통계이며 다중 비교 보정·독립 확인이 아니다.

## 기울기 진단

기존 pi=.05 E/ER, 두 kappa, 다섯 trial의 best/epoch9를 사용한다(40 snapshots).
train의 각 context에서 seed 20261301+trial로 최대256 entities를 비복원 추출한다.
같은 trial/context는 모델 간 동일 subset. context별 mark NLL/base NLL/penalty의
기울기 norm과 cosine을 공유 encoder, copy base, history alpha, route context,
route gap/interaction 및 나머지 블록별로 측정한다. lambda=.01 배율을 명시한다.
이력 h와 raw gap의 reference mean에 대한 penalty-descent 방향 미분도 측정한다.
실제 optimizer step은 없고 checkpoint state/file hash 불변을 확인한다.
기울기 충돌은 국소적 관측이며 학습 이력의 유일한 원인이나 경로 분리의 성공을 증명하지 않는다.

## 외부 비교

SDV 1.38.0 / DeepEcho 0.8.1 CPAR, 공식 기본 예산128 epochs/sample_size1를 사용한다.
전체 train으로 4개 pi × 2 kappa × 5 trials =40회. 기존 고정 wrapper와 전처리를 사용하고
모델·loss history·생성물·시간을 저장한다. sample_size나 epoch를 결과에 따라 변경하지 않는다.
CS-SAF와 같은 seed/정적 context/길이/원본 train entity indices로 2048개 생성한다.
Whole-trajectory train resampling도 같은 계획으로 생성하는 복사 대조군으로만 포함한다.

공통 평가는 기존 generation metric suite를 train에만 fit한 후 validation에 대해
pooled 및 context별로 계산한다. 주요 지표는 transition-conditioned mark TV,
short-gap repeat curve L1, gap-repeat MI error. marginal/lag/length 지표도 전부 보존한다.
CPAR의 연속 gap을 CS bin representative로 강제 투영하지 않는다. likelihood 단위 비교나
CPAR에 존재하지 않는 copy-head oracle TV를 만들어 넣지 않는다. CS-SAF의 oracle TV와
공통 생성 지표는 서로 다른 결과 표로 보고한다. 동일 데이터/조건 비교이지 동일 compute
또는 동일 optimizer 비교는 아니다. CPAR 하나로 모든 외부 sequential 모델 우위를 주장할 수 없다.
공식 참고: https://docs.sdv.dev/sdv/modeling/sequential-synthesizers/parsynthesizer

## 실행·한계

CPU 반복·reload·loss/validity 및 CPAR adapter smoke 후 GPU. 한 GPU당 우리 작업 하나,
매 launch 전 다른 compute process 없는지 확인. 기술 실패는 산출물을 보존하고 신규 dispatch를
멈춘 뒤 원인/수정을 별도 기록한다. 과학적 실패는 나머지 cell 중단 사유가 아니다.
코드·CPU 근거·결과·실패 분석과 교수님용 보고서를 research/cs-saf에 push한다.

기존 seed42 데이터와 validation을 재사용한다. 새 데이터 생성 seed에서의 확인, test,
새 DGP, 실데이터, REaLTabFormer/ARGN 비교는 이 고정 실행 범위 밖이며 완료로 표시하지 않는다.
좋은 결과가 나와도 독립 확인 전 최종 방법 우위/논문 수준의 증거로 주장하지 않는다.
