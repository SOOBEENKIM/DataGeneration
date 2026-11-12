# ARGN 업종–금액 조건부 학습 제거 대조

2026-09-27, 금액 대조8학습/16생성과 후속 실제 이력 진단을 확인한 후 등록한다.
이전 학습의 사전 가설로 소급하지 않는다. 다음 실행은2학습/8자유 생성이다.

## 관측 근거와 선행 연결

balanced_shared의 실제 과거·현재 사기 label에서 현재 필드를 모두 생성하면
사기 금액 log-W1은.822–.831이었다. category만 실제 값으로 제공하면.072–.098로
회복됐지만 gap만/merchant만 실제 값으로 준 대조에서는 비슷한 회복이 없었다.
category는 금액의 앞 조건이며, 이 조건의 생성 오류가 금액 생성에도 전달된다.
이는 현재 설정의 조건부 출력 학습 문제다. 일반 금융 인과관계나 유일 원인 확정은 아니다.
[후속 진단](../argn_amount_learning_v1/condition_probe/PROTOCOL.md).

[ARGN](https://arxiv.org/abs/2501.12012v2)의 기존 필드 조건부 분해,
[CTGAN](https://proceedings.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf)의
조건별 학습 표본 배분, [TabCascade2026](https://arxiv.org/abs/2601.22816v3)의
상류 범주/상태에 조건화한 연속 출력이 가까운 선행이다. 새 필드 의존성을 발명하는
실험이 아니며, 기존 ARGN에서 그 조건부 분포가 충분히 학습되게 만드는 대조다.
큰 새 상태 모듈을 도입하기 전에 동시 보존의 기본 대조를 확보한다.

## 변경과 고정

이전619/69/147고객 split·codec·full history·부모 seed2개를 그대로 쓴다.
기존 금액 balanced_shared checkpoint를 고정한다. 다른 모든 부모 가중치를
동결하고 category의 native regressor/predictor 한 쌍만 복사해 학습한다.
category 입력에는 실제 이력, 고객 context, 앞 구조 필드와 label만 들어간다.
현재 category/gap/merchant/amount는 native permutation mask로 배제된다.
전체 인코딩 위치에서 복제 head/cache logits를 원래 teacher와 대조한다.

정상1024+사기1024 replacement sampling, label별 평균 NLL 각1/2.
Adam.001, native dropout.25, clip1, 최대2000step,100step마다 전체 내부 validation,
macro class NLL 기준 patience5/min_delta1e-5, step0 포함. 금액 학습과 같은 기준이다.
개발 고객으로 checkpoint를 선택하지 않는다. label prior·길이·전이표는 바꾸지 않는다.
각 팔은 추가 학습 대조이며 원래 ARGN 대비 같은 전체 연산 예산의 최종 benchmark가 아니다.

## 네 경우의 제거 대조

모든 경우에 같은 train-only duration label reference, 같은 고객 context,
동일한 fit/generation seed와 native 생성 길이를 쓴다.

| 경우 | category | amount | 산출물 |
|---|---|---|---|
| 기존 기준 | 기존 부모 | 기존 부모 | 기존4생성 보존 |
| 금액만 | 기존 부모 | 고정 balanced_shared | 직전4생성 보존 |
| 업종만 | 새 balanced category | 기존 부모 | 새4생성 |
| 업종+금액 | 동일 새 category | 동일 고정 balanced_shared | 새4생성 |

새 category는 두 경우에 같은 가중치이며 금액 가중치는 기존 실행과 같다.
따라서 두 변경을 따로 적용한 경우와 함께 적용한 경우를 직접 비교한다.
post-hoc category/label/amount 재배정이나 비율 강제는 없다.
category는 모델에서 뽑고 그 값을 native gap/merchant/amount의 조건으로 전달한다.

## 평가와 판단

native 원래 평가의 사기/정상 금액 W1, 개인 rolling 비율 W1, class별
category–amount TV와 gap–category–amount–history TV, merchant/category,
길이·다양성·전이·고객 구성을 모두 기록한다. teacher category NLL/label별 주변 TV도
기록하되 이것만으로 자유 생성 성공을 말하지 않는다.

두 fit seed 평균에서 joint의 fraud amount/relative W1이 amount-only보다 각각
20% 이상 감소하고 normal amount W1 악화가 절대.05 이내인지 practical screen으로
본다. 업종/merchant의 정상 품질 악화도 별도로 공개한다. 모든 draw를 보고하며
두 변경의 수치상 상호작용을 계산해도 두 seed로 통계적 유의성을 주장하지 않는다.
개인별 사기 발생·기간 모델은 고정됐으므로 고객 구성의 해결을 주장하지 않는다.

이 실험이 좋아지면 ARGN에서 희귀 조건부 학습을 정돈한 강한 기준이 된다.
그것만으로 새로운 학습 원리나 2026년 생성기 대비 우위가 확보되지는 않는다.
다음 개인·episode 전환 구조는 이보다 강한 기준과 비교해야 한다.
원래 모델·실패 결과·최종 test 격리를 유지하고 유휴 GPU2/3만 쓴다.
