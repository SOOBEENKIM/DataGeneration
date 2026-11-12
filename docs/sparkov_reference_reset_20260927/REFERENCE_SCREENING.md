# Sparkov 선행 재현 기준 재점검

2026-09-27. 사용자는 새 구조 개발보다 먼저 Sparkov에서 유사한 생성 연구를
찾아 재현할 것을 요구했다. 이번 작업은 문헌·저자 코드 검토와 실행 전 점검이다.
**새 선행모델의 학습·논문 수치 재현을 완료한 기록이 아니다.**

## 먼저 바로잡을 해석

- Berka 공식 구현의 분포 품질 재현과 Sparkov 전체 이력 B/B+S는 다른 실험이다.
  특히 최근 B/B+S는 window 100에서 전체 인코딩 이력으로 변경됐다. 가중치 대조는
  native 거래 손실도 변경했다. 이 결과를 원문 ARGN의 Sparkov 재현 실패와 혼용하지 않는다.
- Sparkov가 잘 안 된다는 이유로 사기 라벨 자체를 원인으로 단정할 수 없다.
  같은 Sparkov, 같은 모델·표현·예산에서 라벨 포함/제외만 대조하지 않았다.
  그런 대조를 한다면 공통 거래 필드의 품질을 비교하며, 라벨 없는 모형에
  사기 지속률 복원을 직접 채점하는 오류를 피해야 한다.
- 현재 S의 근거는 개선 미입증이다. 최종 CS-SAF 연구 전체가 불가능하다는
  결론도, 새 architecture의 필요성·우월성 증명도 아니다.
- 자료 변경과 학습 설정 변경을 충분히 분리하기 전에 구조 기여 후보를 확대하고,
  이 구분을 사용자에게 명확히 전달하지 못한 것은 담당 에이전트의 판단 문제다.

## 검색과 재현 적합성

Sparkov와 CPAR/TimeGAN/CTGAN/TabDiT/FinDiff/TabDDPM, sequential synthesis,
fraud generation, temporal dependencies 등을 조합해 검색했다. 같은 데이터의
탐지 논문, minority 행 증강, 전체 고객 거래열 생성은 별도로 분류했다.
아래는 원문 또는 저자 코드로 확인한 범위이며 모든 문헌의 부재 증명이 아니다.

| 후보 | 자료와 생성 목표 | 공개 근거 및 이번 확인 | 현재 재현 판단 |
|---|---|---|---|
| **VAE-GAN+CPAC**, *Fraud is not just rarity…*, KBS 2026 (2025 preprint) | 저널판 §4.5에서 Sparkov 추가 실험. 사기 행 증강과 탐지 효용 | [출판사](https://doi.org/10.1016/j.knosys.2026.116594), [저자 코드](https://github.com/claudiunderthehood/VAEGAN-CPAC). Sparkov 전용 실행 파일 확인·clone·검사 | 같은 데이터/사기 생성이라는 점에서 보조 후보. ID 제거·행 단위 생성이므로 전체 고객 거래열의 직접 기준으로 대체하지 않음. 공개 코드 점검 항목 아래 기록 |
| **CPAR data-centric**, 2024 | Altman/IBM 카드 자료. 고객별 거래열과 `Is Fraud?` 생성 | [원문 §2.3–3.3](https://arxiv.org/html/2401.00965). 3명 고객·1024 epoch 등 기존 독해와 재대조 | 목표는 더 가깝지만 Sparkov 논문 재현이라고 부를 수 없음 |
| **IFT-GAN**, 2025 | 중국 금융기관 B2C 자료와 UCI 불균형 자료. 시간 인식 사기 증강 | [원문 §IV-A](https://www.researchgate.net/publication/393598387_Integrated_Feature-Temporal_GAN_for_Imbalanced_Transaction_Fraud_Detection). 공개 본문에서 Sparkov 사용·공식 코드 링크를 확인하지 못함 | Sparkov 동일 자료 재현 대상으로 선정할 근거 없음 |
| **TAT-CTGAN**, 2026 | IEEE-CIS. 행 단위 CTGAN 증강 후 시간순으로 재구성해 탐지기 학습 | [출판사 §4.4–5](https://doi.org/10.3390/math14071183). 본문 검색 색인으로 자료·행 생성 단위 확인; 직접 HTML은 429 | Sparkov/전체 거래열 생성 기준이 아님 |
| **Dynamic oversampling-driven KAN**, 2025 | Sparkov 포함, GAN 증강과 탐지기 비교 | [출판사 §3.1.2](https://doi.org/10.1016/j.eij.2025.100712) | Sparkov 생성 관련 보조 문헌. 현재 확인 범위는 순차 전체 거래열 보존을 검증하지 않음 |

HG-PA는 Sparkov 그래프 생성에 관한 저자의 소셜 게시글만 발견했다. 원문·공식
재현 artifact를 확보하지 못했으므로 성능 주장이나 대표 baseline 근거로 쓰지 않았다.

**이번 범위에서 ‘Sparkov + 고객별 전체 거래열 + 정상/사기 관계 생성 + 재현 가능한
공식 실험’의 조합은 아직 확보하지 못했다.** 논문이 없다고 단정하지도,
부분적으로 관련된 탐지 증강 논문으로 그 빈칸을 채웠다고 주장하지도 않는다.

## VAE-GAN+CPAC: 실제 확보·검사한 내용

- 저자 repository commit `b24e6fb3b16d123d580a1e98b6bacd8ff7cc16dc` 고정.
- Sparkov entry point `scripts/vaegan_cpac_fraud_run.py`: `fraudTrain.csv`와
  `fraudTest.csv`, `is_fraud`, seed 42, batch 64, learning rate 1e-4,
  최대 100 epoch, patience 10 설정이 있다.
- `cc_num`을 삭제하고 이웃 거래 이력을 입력하지 않는다. 생성 분기는 학습 사기
  행을 사용하고 CPAC는 정상·사기를 함께 사용한다. 샘플링 함수는 생성 행의 라벨을
  모두 1로 붙인다. 전체 자연 발생 비율이나 사기 시작/종료를 생성하는 방법이 아니다.
- 고정 코드에서 encoder/decoder/CPAC의 random-weight 입출력과 50행 샘플링은
  실행 확인했다. 이는 인터페이스 점검이며 학습·생성 품질 실험이 아니다.

실행 전에 다음 사항을 발견했다. 저자의 결과가 틀렸다는 단정이 아니라,
이 공개 revision을 그대로 실행하여 논문 재현이라고 부르기 전에 해결할 항목이다.

1. plotting 호출은 `max_points_per_class`를 전달하지만 함수는 그 인자를 받지 않는다.
   함수 서명 binding으로 TypeError를 재현했다. 이 호출이 합성 증강 loop보다 앞에 있다.
2. Sparkov entry point는 정규화와 범주 사전을 만든 뒤 train/validation을 분할한다.
   README의 train-only 설명과 비교할 때 내부 validation 전처리 범위 차이가 있다.
3. decoder 출력은 sigmoid [0,1]인데 사용한 robust 정규화는 음수와 1 이상을
   허용한다. 허용된 우리 학습 금액 916,567건에 같은 정규화 함수만 적용하면
   **62.66%**가 이 출력 범위 밖이다. 이는 같은 codec으로 모든 거래 속성을
   재현하는 목적과 맞지 않을 가능성을 보여준다. 원논문의 탐지 점수를 반증한
   실험이 아니며, 범위를 임의 변경한 뒤 원형 재현이라고 부르면 안 된다.

[실제 실행 결과와 소스 해시](CPAC_CODE_PREFLIGHT.json).
전체 학습 모듈 import에 필요한 seaborn은 기존 고정 runtime에 없었다. 환경을
덮어쓰지 않고 정확한 정규화 함수 정의만 AST로 분리 실행했다. 이 때문에도
공식 환경 전체를 재현했다는 주장을 하지 않는다. upstream 코드는 수정하지 않았다.

2026 출판사 페이지는 직접 열기에서 403이었고 검색 색인의 §4.5를 확인했다.
2025 arXiv preprint와 2026 저널판은 같은 범위의 실험으로 취급하지 않는다.
저널판 전체·보충 설정의 독해와 발표 수치 대조가 아직 남아 있다.

## 지금 적용할 실행 순서

1. 새 S 확대·전환 head·diffusion 학습을 보류한다. 기존 실행물은 보존한다.
2. 대표 재현 대상은 자료명만으로 정하지 않는다. 생성 단위·라벨 처리·전처리·분할·
   모델 버전·생성 설정·논문 평가를 실제 코드와 대조한 표가 먼저 있어야 한다.
3. 공개 원형 실행, 원형의 필요한 실행 수정, 우리 데이터/평가 적용, 제안 구조를
   각각 별도 결과로 기록한다. 어떤 수정으로 수치가 변했는지 대응 대조한다.
4. baseline이 모든 관계를 완벽히 복원해야 한다는 조건은 아니다. 원형의 실행
   신뢰성과 비교 조건을 확보한 뒤 남는 오류를 연구 대상으로 삼는다.
5. 라벨 유무를 원인으로 판단하려면 동일 Sparkov 조건의 별도 대조가 필요하다.
   현재 결과만으로 라벨을 삭제하거나 연구 목표를 탐지기로 바꾸지 않는다.

이번 작업의 완료 범위는 **유사 문헌 재검색, Sparkov 저자 코드 확보, 목적·코드
적합성 점검**이다. 선행논문 학습 재현 완료, 새 baseline 성능 확보, 신규성 확보는 아니다.
