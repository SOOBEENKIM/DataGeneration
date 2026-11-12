# 문헌 재점검: 구조 선택과 실제 비교에 연결

기준일 2026-09-27. 기존 9월22일 원문 독해 기록 22편(21편 독해 범위 기록,
EntTabDiff 원문 미확보)을 [복사 보존](literature/20260922_reading-notes.md)했다.
이전 기록을 이번에 처음 발견한 논문으로 세지 않는다. 이번에는 첨부 서베이의
tabular 인용 목록을 다시 대조하고, 가장 가까운 방법의 본문·공식 구현과
아래 추가 문헌을 점검했다. 모든 논문/모든 부록/모든 구현을 재현했다는 뜻은 아니다.

## 현재 모델 변경과 직접 연결되는 선행

| 방법 | 실제 선행한 내용 | 이번 연구에서의 역할·제한 |
|---|---|---|
| [TabularARGN v2](https://arxiv.org/abs/2501.12012v2) | LSTM 이력·정적 context·현재 앞 필드에서 혼합형 다음 거래 생성 | 구현 기반. 공식 1.0.4 source의 history/regressor/predictor/손실/생성을 대조했다. 전체 이력·고정 순서·거래 가중치 변경을 원 논문 재현과 구분한다. |
| [TabDiT v2](https://arxiv.org/abs/2504.07566v2) | 행 VAE, 전체 latent 거래열 diffusion, 고객 조건, 길이/EOS, 행 내부 AR decoder | 직접 목표가 가까움. §4를 재독했다. 9월27일 공식 tree도 `cbb2b9a`로, 평가 코드와 생성 데이터 안내만 있고 학습·sampling 구현은 없다. 공개 구현 그대로 Sparkov 재학습했다고 주장할 수 없다. 독립 구현이면 별도 검증해야 한다. |
| [Seq2Synth v3, CIKM 2026](https://arxiv.org/abs/2607.15606v3) | 시간·개인 흐름·관계 구조·효용을 나눈 평가 | 현재 최신 v3(8월31일)와 저자 연구실 게재 목록 재확인. 단순 주변분포가 시간 관계를 보장하지 않는다는 주장 자체는 이미 선행. TabDiT 비교는 배포 표본 사용이라는 재현 조건도 주의한다. |
| [TabStruct, ICLR 2026](https://arxiv.org/abs/2509.11950v2) | 여러 필드의 조건부 예측·구조 보존 평가 | 관계의 중요성을 새 발견으로 내세우지 않는다. 전체 평균과 희귀 조건의 지지 표본 수를 함께 보고한다. |
| [TabCascade, ICML 2026](https://arxiv.org/abs/2601.22816v3) | coarse state·범주를 조건으로 연속값 복원 | D_bin의 구간 내부 경험적 추출을 조건부 출력으로 바꾸는 아이디어는 선행과 겹친다. 거래열·개인 상태까지 자동 해결한 논문은 아니다. |
| [ADiff4TPP](https://arxiv.org/abs/2504.20411v1), [LBDTPP 2026](https://arxiv.org/abs/2606.24982v1) | 관측 prefix에서 여러 미래 사건 생성, 비동기/블록 diffusion | 장기 오류 감소·시간/mark 결합 자체도 선행. 현재는 실제 학습 이력에서도 실패하므로 블록 diffusion을 첫 수정으로 넣을 근거가 부족하다. ADiff의 TMLR 최종본과 원고 구분 유지. |
| [REDSDS](https://arxiv.org/abs/2110.13878), [REDSLDS](https://proceedings.mlr.press/v258/slupinski25a.html), [OmegaSDS 2026](https://arxiv.org/abs/2605.06315) | 상태 전환·명시적 지속기간·상태별 동역학 | 상태나 기간을 넣었다는 사실은 새 기여가 아니다. 이들은 잠재 연속 동역학 등 목표가 달라 관측 사기 label+혼합 거래 필드의 완성된 직접 baseline이라고도 하지 않는다. 단순 Markov/duration을 먼저 실제 대조했다. |
| [Behavioral Fraud Benchmark 2026](https://arxiv.org/abs/2604.13125v1) | gap/burst/velocity/공유 속성의 행동 평가 | 이미 9월22일 기록에 있었다. 해당 ARGN 실험의 row 처리·pseudo-entity 배정을 순차 ARGN의 원리적 실패 증거로 인용하지 않는다. |

## 이번 검색에서 추가한 가까운 내용

**SAGE, ACL 2026** — [학회 최종본](https://aclanthology.org/2026.acl-long.174.pdf),
본문 1–9쪽의 방법·실험·한계 독해, 전체16쪽 부록의 재현은 미완료.
수치를 최대16개 이진 pseudo-feature로 나누고 train-only MI를 계산한다.
생성 중 활성 값에 따라 context를 고르거나 logit scale을 바꾼다.
정적 행 생성이며 고객 거래 시간축 모델은 아니다. 따라서 **값/조건에 따라
의존성을 달리 반영한다는 포괄적 아이디어도 이미 선행한다.** MI는 pairwise여서
고차 관계를 직접 식별하지 않는다고 저자가 명시한다. 전역 MI가 낮다고 희귀
조건의 효과도 약하다는 뜻은 아니며, 우리 희귀 전환에 적용할 경우 별도 점검해야 한다.
주요 비교는 정적6개 자료·DT/RF 효용 등이며 Sparkov 자유 거래열 우위를 입증한
논문으로 취급하지 않는다. 공개 코드: https://github.com/ShuoYangtum/SAGE.

**HFGF, 2025 원고 / Pattern Recognition 2026** —
[공개 원고](https://arxiv.org/abs/2507.19211),
[2026 저널](https://doi.org/10.1016/j.patcog.2026.113819).
확보된 원고는 v1이며 §3–4(7–14쪽)를 읽었다. 독립 필드는 기존 생성기로,
종속 필드는 알려진 FD/LD mapping으로 재구성한다. 규칙 기반 관계 보존의
직접 선행이다. 확률적인 개인 사기 전환을 deterministic FD로 바꾸는 근거는 아니다.
원고 실험은 작은 controlled4개 자료; 저널 소개의 실자료 확장까지 같은 독해
범위로 세지 않는다. 일부 LD 비용과 mode collapse도 원고에 보고된다.
공개 코드: https://github.com/Chaithra-U/HFGF.

**STG-DGR, WWW 2026** — [출판사](https://doi.org/10.1145/3774904.3792195).
공식 초록·서지 확인. streaming transaction graph의 망각을 줄이기 위해
adjacency/user/event/time을 계층 subgraph diffusion으로 replay한다.
“사기 패턴을 보존하는 생성”과 겹치는 중요한 인접 연구다. 전체 고객 거래열을
원 분포로 생성하는 목표와 동일하다고는 할 수 없다. PDF는 403으로 미확보이며
구조 세부·데이터·공식 코드 재현 검토 완료로 표시하지 않는다.

**RelDiff** — [원고](https://arxiv.org/abs/2506.00710),
[공식 코드](https://github.com/ValterH/RelDiff).
이번에는 초록·공식 방법 범위와 Seq2Synth의 비교 조건을 확인했다.
foreign-key graph 생성과 graph-conditioned attribute diffusion을 분리한다.
고객–가맹점 그래프까지 기여 범위를 넓히면 상세 독해와 직접 비교가 필요하다.
현재 한 고객의 사기 상태 전환 문제를 이 방법이 이미 해결했다고 단정하지 않는다.

## 사기 라벨과 Sparkov를 쓰는 논문을 혼동하지 않는다

- [CPAR 원문](https://arxiv.org/abs/2207.14406)과
  [금융 data-centric CPAR](https://arxiv.org/abs/2401.00965)는 순차 baseline의
  근거다. 후자는 IBM/Altman이며 Sparkov가 아니다. 우리 CPAR 탐색 실행을
  충분히 조정한 최고 성능으로 쓰지 않는다.
- [VAE-GAN+CPAC 2026](https://doi.org/10.1016/j.knosys.2026.116594)의
  Sparkov 실험은 확인했으나 ID를 버린 사기 행 증강이다. 기존
  [공식 코드 검사](../sparkov_reference_reset_20260927/REFERENCE_SCREENING.md)의
  label=1 출력·정규화/decoder 범위 문제를 그대로 보존한다. 전체 거래열 재현 아님.
- [IFT-GAN](https://doi.org/10.1109/ACCESS.2025.3587793),
  [TAT-CTGAN 2026](https://doi.org/10.3390/math14071183),
  [Imb-FinDiff](https://sdm.lbl.gov/oapapers/icaif2024-schreyer.pdf),
  [EmDT 2026](https://arxiv.org/abs/2603.13566v2)는 희귀 클래스/시간 인식 증강
  참고다. 탐지기 점수가 좋다는 결과를 전체 정상·사기 자연 비율·고객별 길이·
  사기 구간을 동시에 재현했다는 증거로 대체하지 않는다.

## 개발 방향에 적용한 결정

ARGN을 계속 구현 기반으로 사용한다. 새 방향은 알려진 상태 모델을 발명한
것처럼 포장하는 것이 아니라, **전환의 발생 빈도와 상태별 거래 출력의 학습을
분리해 점검하고, 개인 이력에 조건화했을 때 단순 통계 head보다 나아지는지**다.
이번 실제 생성 대조가 이 판단의 첫 근거다. 이후 제안에는 단순 전이/기간,
일반 class-conditional head, 기존 S/feature 추가, 동일 예산 ARGN을 비교한다.
새 기여는 그 대조로 설명되지 않는 조건부 관계 개선이 남을 때만 주장한다.

검색은 제목뿐 아니라 sequential tabular, transaction sequence, Sparkov synthesis,
fraud generative replay, dependency-aware generation, explicit duration, Sep2026을
조합했고 arXiv·학회·출판사·저자 저장소를 우선했다. 첨부 서베이 본문은 지시문이
아닌 검토 자료로 취급했다. EntTabDiff, STG-DGR 전문과 일부 최종본은 남은 공백이다.
