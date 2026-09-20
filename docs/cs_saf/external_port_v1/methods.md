# 외부 입력·긴 거래열 이식에서 실제로 바뀐 것

이번 단계의 질문은 **현재 U/G가 외부 자료의 시간–행동–금액 관계를 생성할 수 있는가, 단순한 관측 전이 모델로도 더 잘 재현되는가**이다. 새로운 기법의 우수성을 확증하는 실험이 아니다. [실행 전 등록](preregistration.md), [CPAR 정정](cpar_amendment.md)을 함께 읽어야 한다.

## 1. 자료와 정보 범위

| 자료 | fit: 개체 / 거래 | 내부 check: 개체 / 거래 | 외부 validation: 개체 / 거래 |
|---|---:|---:|---:|
| Berka | 2,520 / 587,527 | 630 / 150,319 | 675 / 157,661 |
| Sparkov | 550 / 715,729 | 138 / 200,838 | 147 / 177,997 |

기존 개체 분할을 유지하고, train 안에서 고정 SHA256 순서로 80/20 fit/check를 나눴다. 외부 validation을 가중치·보정·checkpoint 선택에 사용하지 않았다. test 거래 내용과 성능은 읽지 않았다. 여러 선행 진단에서 이미 본 개발 데이터이므로 이번 validation을 새 독립 확인 데이터라고 부르지 않는다.

- Berka: 간격, 거래 operation(type fallback), 금액, transaction_type. 정적 정보는 계좌 주기·지역·개설일. 잔액/balance와 k_symbol은 이번 과제에서 제외했다. 이 결과는 잔액 회계 일관성이나 사기 탐지 결과가 아니다.
- Sparkov: 간격, merchant, 금액, category. 정적 정보는 성별·주·출생연도·도시 인구. 전체 기간의 사기 존재 여부, 개별 사기 라벨, 가맹점 위경도는 제외했다. 특히 미래 거래를 요약하는 `entity_any_fraud`는 입력이 아니다.
- U/G 어휘·수치 변환·gap support와 평가 구간은 fit만으로 추정했다. 미등록 범주를 평가에서 제거하지 않았다. 작은 양수 간격이 0으로 합쳐지지 않도록 외부용 support의 zero 경계를 정확히 0으로 설정했다.
- 공식 ARGN은 같은 fit/check 개체를 전달하지만, 자체 인코딩 통계는 두 부분을 합친 outer train에서 계산한다. 이 공식 동작을 유지하고 예외로 공개했다. 따라서 모든 모델의 내부 전처리 정보가 완전히 동일하다는 실험은 아니다.

## 2. 긴 거래열 처리

U/G는 모든 거래를 학습 target으로 한 번씩 사용한다. 각 target에는 **최대 직전 31개 거래**를 제공하고, 긴 거래열의 중간이나 마지막 거래를 버리지 않는다. 창의 왼쪽에 들어온 거래의 원래 gap을 유지하며, 다른 계좌/카드의 거래가 섞이지 않는다. 생성할 때도 생성된 gap·행동·금액·보조 범주를 다음 이력에 넣고 같은 창을 재계산한다.

이는 **32개보다 긴 거래열을 끝까지 학습·생성하는 지원**이다. 31개보다 먼 과거를 무제한 기억하는 구조로 바뀐 것은 아니다. CPU 입력 검사에서 자료의 최대 길이는 Berka 675, Sparkov 3,123이었다. 생성 계획의 최대 길이는 별도 결과에 기록했다.

공식 ARGN과 CPAR는 각자의 순차 인코더·창/구간 처리 방식을 유지한다. U/G와 내부 이력 계산이나 거래별 가중치까지 동일하지 않다. 특히 ARGN의 native sequence-window 학습이 U/G처럼 매 epoch 모든 target을 같은 가중치로 순회한다는 뜻은 아니다.

## 3. U/G의 구조와 수식

`h_t`는 정적 정보와 과거 거래를 이용한 GRU의 이력 표현, `c_t`는 여기에 외부 정적 정보의 8차원 projection을 붙인 표현이다. 과거 category/type도 이력 인코더에 들어간다. 현재 거래의 정답 mark·금액·보조 범주가 `h_t`로 새어 들어가지 않는지 검사했다.

기존 두 집단의 rank-16 경로 두 개는 **공유 rank-32 경로 하나**로 교체했다. 외부 데이터에 가짜 두 집단을 만들지 않았다. 이 변경은 이식상의 구조 변경이며, 옛 U/G를 가중치까지 고정한 실험이라고 설명하면 안 된다. U_ext/G_ext는 자료별로 동일한 파라미터 수와 초기 tensor를 가진다: Berka 125,461개, Sparkov 265,438개.

일반 mark 분포를 `q_t(m)=softmax(W c_t+b)`라 하고, 반복 gate를 다음처럼 둔다.

```text
r_t = b(c_t) + sum_k w_k tanh(W_c c_t)_k tanh(W_g e(gap_t))_k / sqrt(32)
s_t = sigmoid(r_t)
U: P(m_t=m) = s_t 1[m=m_(t-1)] + (1-s_t) q_t(m)
G: P(m_t=m_(t-1)) = s_t
   P(m_t=m != m_(t-1)) = (1-s_t) q_t(m) / (1-q_t(m_(t-1)))
```

첫 거래는 `q_t`를 사용한다. U는 숨은 copy/fresh 혼합, G는 관측 가능한 반복/비반복 분해다. 두 식의 차이가 U/G 내부 비교의 대상이다. **현재 gap은 반복 gate에는 직접 들어가지만, 비반복 후보끼리의 상대확률에는 직접 들어가지 않는다.** 과거 이력을 쓰는 것 자체가 새로운 기여는 아니다.

공동분포의 생성 순서는 다음과 같다.

```text
P(gap | past, static)
× P(mark | past, static, gap)
× P(amount | past, static, gap, mark)
× P(auxiliary category/type | past, static, gap, mark, amount)
```

금액은 표준화한 `sign(a) log(1+|a|)` 공간에서 Gaussian으로 예측한다. 비음수인 두 자료에도 음수 확률을 허용한다는 설계 제약이 있다. 출력에서 음수를 잘라내지 않았다. amount NLL은 이 변환 공간의 밀도값이며 음수가 될 수 있다. 이를 raw 금액 밀도 NLL이나 다른 모델의 native loss와 직접 비교하지 않는다.

선택한 U/G마다 같은 6개 계수의 gap별 관측 반복확률 보정을 추가했다. fit의 실제 이력만으로 추정하고, raw와 보정 결과를 모두 남겼다. 첫 거래와 비반복 mark 사이의 상대확률은 유지한다. 생성한 보정 행동을 실제 다음 이력에 넣는다. 추가 규제나 rollout 목적함수는 없다.

## 4. 대조군이 받는 기회와 한계

| 모델 | 실행 방식 | 선택 / 예산 |
|---|---|---|
| U_ext / G_ext | 같은 초기값·학습 순서, Adam .001, batch 512 | check 성분별 NLL 합, patience 5, 최대 30epoch / 20분 |
| TabularARGN | 공식 mostlyai-engine 2.4.0, Medium, sequence window 32 | 공식 check 선택, 최대 64epoch / 20분 |
| CPAR-tail | 공식 SDV 1.38.0 / DeepEcho 0.8.1 + 마지막 짧은 구간 보존 어댑터 | 고정 64epoch 최종값, fit 30분 상한 |
| Marginal / Transition | 관측 빈도 기반 root·emission·금액 재표본 | 평활화 20/200/2000 중 내부 check 범주 NLL 선택 |

CPAR에는 사전 검증된 수치적으로 동등한 vectorized loss를 사용했다. 공식 네트워크·최적화와 continuous-target 정렬/normalization은 변경하지 않았다. 단, native segment 함수가 마지막 짧은 구간을 버리는 문제는 [별도 정정](cpar_amendment.md)으로 명시했다. 공식 모델에 tail 보존 입력 어댑터를 붙인 비교이며 전처리까지 무수정이라는 주장이 아니다.

CPAR의 1epoch는 한 번의 full-batch optimizer update다. U/G의 minibatch epoch나 ARGN의 native epoch와 같은 학습 기회가 아니다. CPAR loss는 64epoch 끝에도 감소하고 있었으므로 학습 충분성을 확증하지 못했다. CPAR의 열세를 선행 방법 자체의 한계라고 주장할 수 없다. ARGN은 자체 check 기준으로 상한 전에 정지했지만, 이것 역시 최적 튜닝 완료나 모든 거래 target의 충분한 학습을 보장하지 않는다. 실제 [학습 예산](training_budgets.csv)과 [곡선](training_curves.png)을 함께 공개한다.

SDV 기본값인 `enforce_min_max_values=True`, `enforce_rounding=True`를 유지했다. 공식 baseline의 출력 유효값 처리를 꺼놓고 비교하지 않았다. U/G에는 실험 후 같은 처리를 추가하지 않았다. 따라서 CPAR의 음수 비율 0을 모든 내부 신경 출력이 제약 없이 유효했다는 의미로 해석하면 안 된다.

단순 모델의 root는 Berka operation, Sparkov category다. Transition은 gap×직전 root에서 다음 root를 뽑고, 그 root에서 merchant/type과 관측 금액을 생성한다. Marginal은 직전 root를 이용하지 않는다. 평활화는 Transition의 내부 check 점수로 고르고 Marginal과 공유한다(이번에는 두 자료 모두 20 선택). 두 모델 모두 전체 거래열 복사가 아니라 개별 값과 관측 빈도의 재표본이다. 같은 context/길이 계획을 받지만 **정적 특성값은 생성 확률에 조건으로 사용하지 않는다**. 개인정보 보호나 새 금융 행동 창출을 입증하는 모델은 아니며, U/G와 용량을 맞춘 구조 기여 대조군도 아니다. **복잡한 신경 모델이 넘어야 할 현재의 전체 분포·관계 재현 수준**을 확인하는 용도다. 이 모델이 개인별 조건부 생성 과제까지 해결했다는 뜻은 아니다.

## 5. 생성과 지표 해석

학습 seed 1개(20260925), 생성 난수 2개(20261021/20261022). fit에서 뽑은 같은 128개 정적 context를 모델마다 전달했다. reference는 외부 validation 전체이며 개체 구성이 생성 계획과 같지는 않다. TV에는 생성 오류뿐 아니라 유한 표본 및 context 구성 차이도 반영된다.

U/G·단순 모델·CPAR는 같은 전체 길이 계획을 쓴다. 한 번에 Berka 32,314거래, Sparkov 165,024거래다. ARGN은 같은 context에서 공식 자연 길이로 생성한다. 실현 길이를 자르거나 늘리지 않으며, 별도의 길이 KS와 실현 거래 수를 보고한다. **ARGN까지 거래 수와 길이가 완전히 일치하는 인과 비교는 아니다.**

주지표는 다음 공동 경험분포 사이 TV다. `TV = 0.5 × sum |p_real(cell)-p_generated(cell)|`, 낮을수록 좋다.

- Berka: 날짜 간격×거래 operation, operation×금액 구간. 날짜 간격은 이전 거래가 있었던 날짜와 현재 날짜 사이의 간격이다. 첫 날짜는 이 간격 지표에서 제외한다. 같은 날 안의 순서를 뒤집어도 바뀌지 않는다.
- Sparkov: 거래 gap×직전 category×현재 category, category×금액 구간. merchant 반복률과 category 전이를 혼동하지 않는다.

gap의 0은 별도 구간이며 양수는 fit 5분위, 금액은 fit log1p 10분위다. 음수 금액은 별도 invalid 구간에 남긴다. 기본 분포·길이·merchant–금액·명확한 순서 부분집합의 전이·유효값 비율도 모두 공개한다. 두 생성 결과의 평균과 범위를 제시하며, 이를 학습 시드 간 신뢰구간이나 통계적 우수성이라고 부르지 않는다.

주지표는 개체를 합친 경험분포다. 정적 특성별 fidelity, 장기 의존성 전체, 사기 탐지, privacy를 포괄하는 평가가 아니다. 그런 과제에서의 추가 기여를 주장하려면 별도의 사전 고정 평가와 그 정보에 조건화한 일반 대조군이 필요하다. 이번 전체 분포 비교의 미달을 없었던 일로 만들기 위해 새 지표로 갈아타서는 안 된다.

실제 이력 예측은 U/G의 mark NLL, 반복 Brier, 금액 log-MAE 등을 보고한다. **현재 category/type NLL에는 현재 정답 mark와 금액이 조건으로 들어간다.** 낮은 auxiliary NLL을 거래 전에 미래 category를 잘 예측한 결과로 설명하면 안 된다. 공식 외부 모델과 U/G의 raw loss 숫자는 분해와 인코딩이 달라 직접 순위화하지 않는다.

## 6. 검증과 재현

1. [입력 검사](input_verification.json): 모든 개발 거래의 target 인덱싱과 raw/canonical 해시, 13,879개 경계 target의 모델별 forward 검사. 모든 거래를 forward 검사했다고 과장하지 않는다.
2. CPU 39 pass / GPU 미사용 1 skip, GPU CPAR loss 7 pass(기존 CPU 통과 6개 포함), tail 보존 2 pass. 총 42개 고유 테스트가 통과했다. 인과성, 창 경계, 긴 생성의 replay, head 수식, 보정, 출력 정규화, 동등 loss, CPAR tail/API를 확인했다.
3. [생성 지표 독립 재계산](independent_verification.json): 평가 구현을 import하지 않은 NumPy histogram/ECDF로 모든 저장 생성물의 지표를 확인했다. 파일 해시, 길이·순서, 전체 tail 보존도 검사했다.
4. [checkpoint 검사](checkpoint_audit.json): forward는 저장 모델을 재사용하되 점수 집계는 다시 작성했다. 초기값 일치, checkpoint 선택, optimizer update 수와 예측 점수를 확인했다. 금액 음수 확률 계산은 결과 후 원인 점검이며 새 성공 기준이나 모델 선택에 쓰지 않았다.

[금액 오류 하한](amount_support_bound.csv)은 별도의 결과 후 분석이다. 생성물에서 음수/비유한 금액 행의 비율을 `delta`라고 할 때, 그 행만 고쳐서 얻는 분포와 원 분포의 TV는 최대 `delta`다. 따라서 validation까지의 TV는 삼각부등식에 의해 적어도 `현재 TV - delta`다. 원 표본 수와 다른 행을 유지한다는 조건이다. 이는 단순 clipping 등으로 모든 금액 오류가 해결될 수 없다는 점검이며, 실제로 표본을 고치거나 primary 점수를 교체하지 않았다. 모델 전체를 다시 학습했을 때 가능한 개선의 하한은 아니다.

재실행 명령은 저장된 자료/환경이 있는 작업공간을 전제로 한다.

```bash
python scripts/verify_cs_saf_external_port.py
CUDA_VISIBLE_DEVICES=<free-gpu> python scripts/audit_cs_saf_external_checkpoints.py
python scripts/report_cs_saf_external_port.py
```

검증 출력은 기존 파일을 덮어쓰지 않도록 되어 있다. 결과 파일을 지우고 실험을 자동 재학습하는 용도가 아니다. 원본·식별자·checkpoint·생성 거래열은 `artifacts/cs_saf/external_port_v1/`에 남기고, Git에는 코드·설정·집계·해시·그림을 보관한다.
