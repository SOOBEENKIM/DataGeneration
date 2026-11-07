# 초기 시간 출력과 반복 입력 보정의 연결 — 실행 전 고정 (2026-09-28)

사용자는 최신 후보의 초기 속도 형성과 후속 보정 연결을 수정하고 검증하도록 승인했다.
현재 비교 기준은 stratified_loss_v1/personal_six이며 활성 clock_prefix도 보존한다.
초기 진단의 oracle 수치를 생성 성능으로 사용하지 않는다. 새 모델·가중치·원시 결과는
별도 ignored artifact에 보존하고, 개선되지 않으면 기존 모델을 유지하며 push하지 않는다.
최종 test는 열지 않는다.

## 가장 가까운 선행과 남은 문제

[Shchur/Biloš/Günnemann, ICLR2020 §3.2–3.3](https://arxiv.org/pdf/1909.12127)는
간격의 log-normal mixture와 이력/metadata/sequence 조건부 분포를 제시한다.
이번 초기 출력은 log1p 간격의 3성분 Gaussian mixture를 학습해 기존 정수 간격
sampling으로 변환하는 단순 통계 대조다. 논문의 전체 신경모델 재현이나 새로운 mixture
방법이 아니다. 고객 조건을 넣기 전에 초기 marginal 적합만으로 해결되는 범위를 확인한다.

[Zhang et al., LBDTPP, 2026 §IV-C/E](https://arxiv.org/html/2606.24982v1)는
빈 이력에서 시작하는 생성과 후속 prefix 조건 생성, prefix 안정성 가정 아래의 오차
전파를 다룬다. 초반 정확도와 이후 조건 분포를 함께 보아야 한다는 점이 겹친다.
그 논문의 잠재 블록 diffusion/이론 보장은 여기서 구현하거나 주장하지 않는다.
블록 생성/초기 상태/누적오차 완화 자체는 신규 기여로 잡지 않는다.

현재 진단에서 실제 clock으로만 바꾸면 6–49 오류 .311→.064, 첫5 실제 memory면
.086으로 줄었으나 100–199 비용이 있었다. 따라서 초기 marginal 출력만 변경한 경우와
그 뒤의 보정을 생성 이력에서 맞춘 경우를 분리하는 2×2 대조를 한다.

## 변경과 비교

| 팔 | 초기 normal-stay, index1–5 | index6–199 추가 보정 |
|---|---|---|
| current(기존 저장 결과) | 현재 GMR+보정 | 없음 |
| init_only | 새 초기 mixture | 없음 |
| handoff_only | 현재 GMR+보정 | 반복 생성으로 학습한 2계수 |
| init_handoff | 새 초기 mixture | 같은 방법으로 학습한 2계수 |

초기 mixture는 optimization의 previous=0,current=0,index1–5만 사용한다.
sklearn GaussianMixture K=3,n_init=3,seed20261206,reg_covar1e-4,max_iter300.
전체 고객의 실제 과거를 복사하지 않고 새 간격을 확률적으로 생성한다. 유효 개인 속도가
없는 구간의 단순 모델이므로 그 안의 개별 고객 관계를 보장하지 않는다.

추가 보정은 log1p(position)의 6/50 두 hat 계수와 200의 고정0 끝점이다. index<6,
index>=200에서는 정확히0이다. 각 계수±.35, 기존 보정과 합한 후 기존처럼±.35로 clip.
현재 personal_six의 6계수, GMR weights/means/covariances, clock endpoints는 동결한다.
normal-stay만 변경한다. event0의 native 출력, 다른 사기 상태의 조건부 법칙, clock
갱신 규칙, label/count/category/amount head는 그대로다. 생성된 이력이 달라지므로
나중 시간·금액 결과까지 같다고 가정하지 않는다.

두 handoff 적합 모두 실제 label/length를 고정하되 **시간 과거는 전부 자체 생성**한다.
고객별 전체 길이(1024 제한 없음)에서 기존 stratified 목적을 사용한다. 동일한 고객
가중치와 200+ 세분화 질량을 유지한다. 기존 수식의 clock 항은 이제 생성 clock과 실제
clock의 거리를 평가한다(기존 actual-prefix에서는0). 개인/사기 onset 관계는 capped
clock이 아닌 기존 raw strict-past median 분모로 평가한다. 기존 hinge 허용값도 그대로다.
두 팔의 hinge reference는 모두 현재 frozen 모델의 같은 전체 반복 재생이다.

2계수 zero 초기화, Powell,xtol/ftol .001,maxfev192(초기 평가 포함),seed20261201.
훈련 목적 best-observed를 선택한다. 내부 검증69명×2noise는 결과 진단만 하며 선택/튜닝에
사용하지 않는다. 초기 mixture1적합+보정2적합, native ARGN 새 학습0회다.
학습 조건부 경로와 전체 자유 생성 결과를 구분한다. 개발147명은 학습에 쓰지 않는다.

## 검증과 실제 전체 생성

- NumPy/Torch 모수 일치, 6/50/200 경계, mixture 정규화/양의 분산, identity 출력,
  비대상 phase와200+의 같은 조건부 출력, 원래 자체clock 재생의 첫200 exact parity 확인.
- 첫 gap 유지 및 batch 축소/재정렬은 각 native worker의 기존 통합 검사로 재검증.
- 2개 frozen 부모×4개 기존draw×3새팔 = **24개 GPU 전체 거래 생성**.
- 각 draw label/length 동일성, native checkpoint 및 frozen 모듈 SHA256 확인.
- 외부 작업 없는 GPU만2회 확인해 사용하고 장치당 우리 worker 최대2개. 외부 작업 중지 없음.
- 독립 기존 evaluator로 시간/금액/사기/가맹점/다양성 재계산. 1–5,6–20,21–49,
  6–49,50–99,100–199,50–199,200–499,500–1023,1024+,2048+도 같은 고객 길이 지원에서 확인.

## 판정 — 실행 후 변경하지 않음

각 부모에서 current 대비 6–49 평균10% 이상 감소,4draw중3개 이상 개선이 일차 목표다.
전체 정상 시간은 current+.005 이내,1024+는+.02 이내,고객균등 정상은+.01 이내,
50–199/100–199/200–499/500–1023는+.03 이내를 현재 주요 개선 보존 조건으로 둔다.
onset 개인+.01,정상 개인+.005,사기gap/개인사기금액/정상금액/정상merchantTV+.01,
가맹점 다양성95% 이상을 동시에 확인한다. 비용의 실제 크기를 함께 보고한다.
label/length가 같으므로 전체 사기율은 matched reference와 같아야 한다.

활성 clock_prefix 교체 조건은 기존 보수적 screen도 별도로 그대로 적용한다.
현재 후보 대비 개선과 활성 기준 교체를 혼동하지 않는다. 혼합 결과는 로컬 후보로 남기고
push/자동교체하지 않는다. 두 단독 팔보다 결합이 낫지 않으면 결합 필요성/새 기여를
주장하지 않는다. test/다른 데이터/최신 생성기 직접 비교는 이번 연구 범위 밖으로 남는다.
