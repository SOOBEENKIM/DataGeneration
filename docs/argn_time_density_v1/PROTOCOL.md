# 단일 조건부 시간 출력: 실행 전 등록 — 2026-09-27

## 검증할 주장과 선행의 경계

새 구조의 기여를 확정하지 않는다. 기존 경계 head는 개인 대비 onset gap에서
phase GMM보다 낫지만 전체 사기 gap을 악화시켰다. GMM은 전체 gap을 잘 맞추나
정상 개인 gap을 악화시켰다. 단일 조건부 밀도에서 두 관계를 함께 보존할 수
있는지를 검증한다. 복귀의 상대 간격 우위는 추가 seed에서 반복되지 않았으므로
주요 주장은 **사기 시작 시 개인 거래 속도 변화**로 한정한다.

- [Waghmare et al., CIKM2022, §3.2 식5–11](https://arxiv.org/html/2210.15294v2):
  history 및 현재 mark 조건부 lognormal mixture가 직접적인 출력 선행이다.
  조건부 mixture 자체를 새 기여라고 하지 않는다.
- [MoTPP, 2026-09-18, §4–5](https://arxiv.org/html/2609.21382v1):
  zero-inflated lognormal 출력이 이미 있다. 여기서는 1초 정수 codec의 각 구간
  확률을 적분해 0초와 양 끝값을 같은 밀도로 학습한다. 별도 zero head는 넣지 않는다.
- [RevIN, ICLR2022 원문](https://openreview.net/pdf?id=cGDAkQo1C0p):
  개인 통계를 사용한 가역 정규화/복원 자체는 선행한다. 여기서는 미래 구간의
  통계가 아닌 strict-past 20간격 median을 사용한다. 이것만으로 신규성은 없다.
- [On the Role of Reversible Instance Normalization, 2026, §2–4](https://arxiv.org/html/2603.11869v1):
  정규화로 조건 정보가 사라지는 문제가 있어 원래 frozen 이력 표현과 clock을
  입력에 유지한다. 해당 논문의 예측 설정과 거래열 자유 생성의 차이도 구분한다.
- [LBDTPP2026](https://arxiv.org/html/2606.24982v1),
  [OmegaSDS2026](https://arxiv.org/html/2605.06315v1)는 현재 공식 코드/원문
  검토를 보존한 가까운 생성 선행이다. 이번 같은 용량 출력 대조를 이 모델들의
  Sparkov 직접 재학습 완료라고 부르지 않는다. 최신 생성기 전체 직접 비교는 남는다.

## 동결과 단 하나의 변경

기존 onset_fit 2개 부모(20260930/20261001)의 ARGN, label/count, 업종/금액
가중치를 보존한다. 첫 이벤트에는 gap 정의가 없으므로 새 head를 적용하지 않는다.
나머지 전체 gap을 하나의 3성분 log1p-time Gaussian mixture로 생성한다.
새 gap이 이력과 이후 금액에 영향을 줄 수 있으므로 모든 품질 비용을 평가한다.

| arm | 입력 | 출력 좌표 |
|---|---|---|
| history_only | frozen gap 직전 이력/현재 선행 필드 + 전환 one-hot | log1p(gap) |
| history_clock | 위 입력 + clock, 유효성 | log1p(gap) |
| history_relative | history_clock과 동일 | log1p(gap) − clock, 샘플 시 clock 복원 |

clock=log1p(strict-past 20간격 median), 관측 최소5개. 각 과거 gap을 7일로
상한 처리한다(기존 numeric decoder의 보호 tail보다 낮음). 부족/median0이면
optimization 양수 간격 median을 사용하고 유효성0을 준다. 이 대체값도 학습 자료만
사용한다. 평가는 기존 uncapped real/generated gap 기준을 유지한다.
첫 gap은 clock에 넣지 않는다. 생성 시 생성된 값만 사용하고 고객 state에 clock
ring을 실어 batch 축소/순서 변경에도 보존한다.

세 arm 모두 입력차원·64 hidden SiLU·3성분 출력과 파라미터 수가 같다.
history_only의 clock 두 채널만0이다. frozen base의 평균/표준편차는 optimization에서만
계산한다. 각 전환(0→0,0→1,1→0,1→1)별 GMM 초기 prior는 해당 좌표로 optimization
자료만 사용해 계산한다(최대50000점, seed20261115,3성분). NN 최종층0으로 시작한다.
relative prior는 좌표가 달라 초기 분포도 다르며 동일 초기 함수 실험은 아니다.
유효성 입력이 있는 일반 history_clock을 가장 중요한 같은 정보 대조로 둔다.

## 학습과 실행

시간 단위/범위는 frozen codec(1초,0–1045691)이며 NLL은 [gap,gap+1)의
정확한 혼합 정규 CDF 차이다. 양 끝 코드는 외부 tail을 포함한다. 별도 truncation
정규화나 연속 density를 정수 likelihood라고 부르는 오류를 피한다.

부모별 arm3개=작은 head6학습. AdamW lr3e-4, decay1e-4, batch512(4전환 각128),
최대2000step,100step마다 내부 validation의 4전환 NLL 평균으로 선택한다.
step0 포함, patience6,min_delta1e-5,gradient clip1. 모든 arm의 sampler/예산 동일.
실제 과거 진단은 optimization/internal/development에서 각4 sampling seed
20261111–14. 자유 생성은 2부모×3arm×4draw(20261011/12/21/22)=24개다.
추가 baseline GPU 확인8개는 이전 가중치/프로토콜 그대로 별도 실행한다.
CPU 확인값을 GPU 결과에 섞지 않는다. 최종 test 이벤트는 읽지 않는다.

native cache logit replay, clock의 학습/생성 일치, 첫 gap 제외, 축소/순서변경,
정수 확률 총합/극단 tail gradient, 기존 모델 strict checkpoint load를 검사한다.
각 자유 생성 전체 label열·고객 길이가 대응 onset_fit 결과와 정확히 같은지 검증한다.
모델·코드·자료 hash를 학습 전에 동결하고 실패/원본을 보존한다.

## 사전 판단 기준

주지표: 사기 onset에서 log1p(gap / strict-past20 gap median)의 W1.
기존 평가 정의(최소5, 양수median)를 유지한다. 각 부모4draw 평균을 따로 보고한다.
같은 GPU 결과의 phase_mixture와 history_clock보다 **두 부모 모두** 감소해야
relative 표현의 추가 효과 후보라고 부른다. 8draw를 독립 학습8회로 취급하지 않는다.

각 부모에서 frozen 대비 비용 screen: 개인 대비 사기 금액/정상 금액/정상 gap/
정상 merchant TV/정상 개인 gap W1 증가≤.02, 전체 사기 gap 증가≤.05,
고객당 merchant 다양성 감소≤5%. 추가로 phase_mixture 대비 전체 사기 gap 및
정상 gap 증가≤.03를 요구한다. 목표와 비용을 모두 통과해야 다음 후보로 올린다.
이는 탐색용 practical screen이며 통계적 비열등성의 증명은 아니다.

history_clock으로 해결되면 알려진 조건부 밀도 효과로 분류한다. relative가 전체
시간/개인 관계를 함께 개선하지 못하면 그 표현 가설을 기각하고 기존 후보를 유지한다.
같은 자료의 반복 개발 한계, 다른 dataset/최신 생성기/최종 test 미실행을 명시한다.
