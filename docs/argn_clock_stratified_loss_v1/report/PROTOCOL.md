# 200+ 손실 구간만 세분화하는 대조 — 실행 전 고정

사용자가 요청한 손실 분해 실험이다. 기존 clock_prefix와 HEAD, 이전 실패 결과를
보존한다. 새 파일은 이 ignored artifact 안에 둔다. 최종 test는 읽지 않는다.
결과 개선을 보장하지 않는다. 새 구조나 최신 모델 대비 우위를 주장하지 않는다.

## 근거·선행·검증 가설

직전 adaptive_v1에서 position_six는 전체 정상 gap W1 .18852, 개인 속도를 반영한
personal_six는 .17002였다. 보존한 clock_prefix .08397보다 나빴다. 1024+ 정상 W1도
.33332에서 각각 .41300/.50151로 악화됐다. 기존 목적은 [1,50),[50,200),[200,H)의
세 위치 marginal을 평균한다. 200+ 안의 다른 오류가 pooling에 가려질 수 있다.
실제 이력을 제공한 내부 진단에서도 2048+ 오류가 남아 생성 누적오차만으로 설명되지 않았다.
다만 개발평가 2048+는 평균4.75고객으로 적다. 꼬리 구간의 결과를 전체 고객으로 일반화하지 않는다.

가장 가까운 이론은 Chemseddine/Hagemann/Steidl/Wald의
[JMLR26,2025](https://www.jmlr.org/beta/papers/v26/24-0586.html),
[공개v3 §3 Proposition1, §4](https://arxiv.org/html/2403.18705v3)다.
조건을 바꾸지 않는 coupling의 비용이 조건별 Wasserstein의 기댓값과 같다는 결과를 읽었다.
여기서는 금융 생성 보장이나 논문의 Bayesian OT flow matching을 구현하지 않는다.
이산 위치별 W1의 가중합이라는 알려진 특수 경우를 원인 대조에 사용한다.
기존 연구가 이미 조건부 분포 보존을 다루므로 이 손실 세분화 자체는 신규 기여가 아니다.

최신 시간 생성 문맥은 [Zheng et al., JASA2026](https://doi.org/10.1080/01621459.2026.2657051),
[공개v2 §2](https://arxiv.org/html/2407.03774v2)의 주변 시간분포와 lag 의존성 분리다.
해당 정상성 조건은 우리 비정상 고객 거래열에 그대로 성립하지 않는다. 최신 연도만을
이유로 이론을 차용하지 않고 이번 손실 변경에는 위 조건부 Wasserstein 원문을 직접 연결한다.

가설: **기존 row measure를 유지한 조건부 손실이 실제-prefix와 자유 생성의 후반 오류를
함께 줄일 수 있는가.** 학습 손실 하락만으로 성공이라 하지 않는다.

## 정확히 하나의 개입

기존 L = .5*(W_normal + (W_1:49 + W_50:199 + W_200+)/3)
             + .25*W_clock + 기존 personal/onset/fraud hinge.
새 L은 위 W_200+ 하나만 sum_g alpha_g W_g로 바꾼다.
g=[200,500),[500,1024),[1024,2048),[2048,H).
alpha_g = 해당 구간이 기존 200+ row measure에서 갖던 질량이다.

- event_two: 기존 행 가중치1을 사용, alpha=구간 행수/200+ 행수.
- 다른 세 팔: 기존 200+ 안의 고객별1/N_customer(200+)를 그대로 각 구간에 제한한다.
  **구간별 고객 빈도를 다시 계산하지 않는다.** alpha=제한된 가중치 합/원래 가중치 합.
- 초기 두 구간의 비중, 200+ 전체 비중1/6, 전체 정상 항1/2, hinge의 계수/허용치 불변.
- 각 원본 행과 고객의 총 질량이 세분화 전후 동일함을 검증한다.
- 다른 모든 항은 같은 theta/같은 noise에서 기존 코드와 수치적으로 동일해야 한다.
- 동일 actual-prefix 조건에서 조건부 W1 >= pooled W1이지만 이것은 생성 품질 개선 보장이 아니다.
- 여섯 구간을 단순 균등 평균하면 후반 상대 가중치가 바뀌므로 사용하지 않는다.

## 고정 설정과 비교

이전 네 팔 event_two/customer_two/position_six/personal_six 각각의 동일 구조에서
coarse→refined 손실만 비교한다. 서로 다른 팔 사이 비교는 기존에 존재한 구조/가중치 차이를
포함하므로 이번 손실의 순수 효과는 같은 팔의 전후로 해석한다.
4개 시간 보정 적합, 기존2부모×같은4draw×4팔=32개 새 전체 거래열 생성.
32독립 모델 학습이 아니며 native ARGN 재학습0회.

기존 backbone/GMR buffers/category/amount/label/count, normal-stay에만 delta를 더하는
구조, delta±.35 clip, 두계수/여섯계수 basis와 bounds, 학습자료 전체619고객815664행,
seed20261201, strict-past features, clock endpoints, zero 초기화, Powell xtol/ftol .001,
maxfev1536(초기점 포함), best-observed 선택 모두 adaptive_v1과 같다.
clock endpoints는 이전 값을 복사한다. 내부69고객×2noise는 진단만 하며 재튜닝하지 않는다.
후반 희소성 때문에 새로운 최소표본/병합/균등 질량 조정도 이번에는 하지 않는다.

## 검증·자원·결정

원래 row mass 보존, pooled 오류0/conditional 오류양수인 toy, collapsed partition parity,
SciPy weighted W1, 다른 손실항 exact parity와 차이=pooling_gap/6을 먼저 검증한다.
원형 source/이전 checkpoint/input SHA를 고정한다. 같은 draw의 라벨·길이 exact parity,
원시 파일의 독립 W1·금액·업종·다양성·위치별 지원 수를 검증한다.
GPU는 외부 compute 작업이 없는 장치만 두 차례 확인하여 선택한다. 그 장치에 우리 worker를
최대4개 배치할 수 있고, 추가 직전 외부 점유를 재확인한다. 외부 작업은 중지하지 않는다.

주요 해석: 같은 팔에서 (1) actual-prefix 후반 손실/구간 오류와 (2) free-generation 전체
정상/1024+ 오류가 어떻게 달라지는지, 두 부모와 네 draw에서 방향이 반복되는지 기록한다.
기존 clock_prefix 교체에는 이전의 보수적 screen을 그대로 쓴다: 각 부모 전체 정상5%이상
감소 및4draw중3개이상개선, 1024+ 및 고객균등 정상 악화없음; onset 개인+.01, 정상 개인+.005,
사기gap/개인사기금액/정상금액/정상merchantTV 각각+.01, 다양성95%이상,
matched6–49/50–199/200–499/500–1023 오류+.03이내.
이 기준은 자동교체 기준이며 논문 가능성의 필요충분조건이 아니다.
악화/혼합이면 이전 모델을 유지하고 commit/push하지 않는다. 실패 기록은 로컬 보존한다.
