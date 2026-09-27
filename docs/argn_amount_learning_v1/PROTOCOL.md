# ARGN 희귀 상태별 금액 학습 대조 — 2026-09-27

분류: ARGN 개발의 첫 학습 대조. 알려진 조건부 학습/전문가 분리를 적용하며
새 방법의 신규성을 주장하지 않는다. 사용자 승인 아래 실행하며 결과를 보기 전에 고정한다.

## 근거와 가까운 선행

앞선 실제 이력 진단에서 학습 사기 금액 중앙값408.43에 대해 약51을 출력했다.
ARGN의 amount는 이미 label/category/gap/merchant를 조건으로 받는다.
따라서 새로운 정보나 표현을 동시에 추가하지 않고 학습 비중과 공유 구조를 먼저 대조한다.

- [ARGN v2 §3.5–3.6](https://arxiv.org/abs/2501.12012v2): 이력·고객·앞 필드로
  subcolumn을 순차 생성한다. 공식1.0.4의 amount regressor/predictor를 그대로 복사한다.
- [CTGAN NeurIPS2019 §4.3](https://proceedings.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf):
  조건부 학습과 training-by-sampling은 기존 방법이다. 이번에는 GAN이나 CTGAN의
  log-frequency sampling을 재현하는 것이 아니라 조건부 likelihood의 가중치를 대조한다.
- [Imb-FinDiff2024](https://sdm.lbl.gov/oapapers/icaif2024-schreyer.pdf): 희귀 클래스
  생성 보강 자체는 기여가 아니다. 전체 거래열의 자연 사기 발생률은 별도 문제다.
- [TabCascade2026](https://arxiv.org/abs/2601.22816v3): 조건에 따른 연속값 복원의
  직접 선행. 이번에는 digit 표현을 유지하므로 연속 분포 구조를 동시에 도입하지 않는다.
- 기존 [최신 문헌 대조](../research_reaudit_20260927/LITERATURE_UPDATE.md)의
  SAGE/상태·기간 모델과의 중복 판단도 유지한다. 최신 검색을 재확인했으며
  정적 minority augmentation을 고객별 자유 거래열 생성과 같은 과제로 취급하지 않는다.

## 데이터와 학습 범위

같은619고객/816,283 optimization 거래,69고객/100,226 internal-validation 거래,
147고객/177,997 development 거래. 최종 test 거래는 열지 않는다.
두 기존 B_event_label_first checkpoint(20260930,20261001)를 각각 부모로 사용한다.
같은 전체 이력·codec·column order·고객 context를 유지한다. 공식 window100 대비
장기 이력 우위를 주장하는 실험은 아니다.

이력 encoder, context, 모든 embedding, label/category/gap/merchant/길이 출력을
동결한다. 실제 과거의 amount 이전 regressor 입력을 float32로 cache하고,
실제 amount의 앞 digit embedding은 기존 모듈로 재구성한다. cache에는 현재
정답 amount column 전체가 들어가지 않는다. 다음 digit에 앞 digit을 주는 것은
원래 autoregressive teacher forcing이며 sampling에서는 생성한 digit을 넣는다.
모든 split의 replay logits와 원래 teacher logits를 대조한다.

## 사전 고정한 네 학습 팔

1. **natural_shared:** 기존 amount head 하나. 자연 거래 비중의 conditional NLL.
2. **balanced_shared:** 같은 head 하나. 정상/사기 conditional NLL 평균에 각1/2.
3. **balanced_mixture:** 두 동일 head의 예측 확률을1/2씩 섞는다. 두 head 모두
   모든 label의 자료를 학습한다. 출력 용량 증가의 대조다.
4. **balanced_routed:** 같은 두 head를 현재 정상/사기 label로 선택한다.
   mixture와 정확히 같은 파라미터 수·복제 초기화·최적화 조건을 갖는다.

두 전문가의 초기값은 부모 head와 같다. shared도 같은 부모에서 시작한다.
기존 frozen parent는 별도 무학습 대조로 보존한다. residual adapter나 새로운
개인 상태 입력은 아직 넣지 않는다. routed의 개선만으로 새 기여를 주장하지 않는다.

네 팔 모두 매 step 정상1024+사기1024를 replacement sampling한다. 같은 seed에서
같은 표본 index stream을 쓴다. natural은 두 클래스 평균에 원래 train 빈도를 곱해
자연 비중 NLL의 unbiased estimate를 쓰고, balanced는 각1/2을 곱한다.
label head/prior는 재학습하지 않는다. amount 조건부 학습을 균형화하는 것이
전체 생성 label을50:50으로 강제하는 것은 아니다. 생성 금액은 이후 이력에
들어가므로 다른 거래 속성에 미치는 간접 영향은 자유 생성으로 평가한다.

Adam lr.001, dropout.25, gradient norm1, batch2048, 최대2000step.
100step마다 전체 internal-validation의 label별 amount digit NLL을 구한다.
네 팔 모두 두 클래스 평균을 checkpoint 기준으로 쓰며 patience5/min_delta1e-5.
초기 step0도 후보에 포함한다. 따라서 목적 가중치 외 선택 기준 차이는 없다.
추가 학습이므로 원래 전체 모델의 end-to-end 재학습과 같다고 주장하지 않는다.
모델별 실제 선택 step·업데이트 수·시간·파라미터 수를 공개한다.

## 평가와 다음 판단

각 팔/부모에서 optimization와 development의 모든 사기+같은 수의 고정 추출 정상
위치에 대해 실제 이력·앞 필드를 주고 amount만4회 생성한다. 같은 sampling 코드와
seed로 부모도 재평가한다. normal/fraud의 median,p90,log1p Wasserstein을 보고한다.
DIGIT 직접 복원 평가이며 native 보호 tail 처리까지 포함한 무조건 생성과 구분한다.

학습한4팔×부모2개×생성seed2개의 **16개 자유 거래열**을 모두 생성한다.
앞선 duration label reference를 모든 팔에 동일하게 연결하고 자연 길이를 생성한다.
기존 frozen ARGN+duration의4개 생성물을 직접 대조한다. 사후 label 수정·비율 강제·
금액 교체는 하지 않는다. amount는 native digit sampling 안에서 생성한다.
원래 평가에 따라 사기/정상 금액, 개인 rolling 금액 비율, category/merchant/gap,
고객 구성, 구간 길이, 다양성·유효성을 같이 기록한다.

개발 진행 기준은 p-value나 논문 성공 기준이 아닌 **다음 단계의 실용 screen**이다.
두 fit seed 모두에서 balanced 팔의 개발 conditional fraud log-W1이 같은 코드의
부모보다20% 이상 감소하고 normal log-W1 악화가 절대.05 이하인지 본다.
자유 생성에서는 동일 seed의 frozen duration보다 fraud amount/log-relative W1이
각각 개선되는지, 정상 amount W1 악화가 절대.05 이하인지 함께 본다.
모든 결과를 보존하며 하나의 draw만 고르지 않는다. 평균이 좋아도 고객별 구성
해결이나 전체 관계 보존 성공으로 확대하지 않는다.

- shared 균형화로 충분하면 학습 보정 결과로 채택하고 routing의 신규성은 폐기한다.
- 같은 용량 mixture를 넘는 routed 이득이 남으면 파라미터 공유 효과의 후보 근거다.
- 실제 조건에서는 나아져도 자유 생성에서 실패하면 생성된 조건/이력 이동을 다음
  분석 대상으로 둔다. 이 경우 큰 상태 모듈을 곧바로 추가하지 않는다.
- 출력 회복이 확인되면 다음 단계에서 개인·episode 전환과 출력의 결합을
  전환만/출력만/둘 모두/개인정보 제거 대조로 진행한다. 그 학습 조건은 별도 등록한다.

이번 실험은 금액 출력만 다룬다. merchant/gap 조건부 분포의 학습 보정이나
최신 생성기 전체 대비 우위를 완료했다고 부르지 않는다. 데이터·source·부모 가중치
hash를 고정하고 실패한 실행도 보존한다. GPU2/3만 두 번 idle 확인 후 사용한다.
