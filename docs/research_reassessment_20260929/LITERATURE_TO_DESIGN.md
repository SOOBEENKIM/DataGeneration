# 원문 재검토 → 설계·비교 결정

2026-09-29. 이 표는 제목 목록이 아니라 중복되는 방법, 우리 적용 범위, 다음 대조를 구분하는 기록이다. `SOURCE_INDEX.csv`의 24개 원문은 로컬 HTML/TXT로 보존했고 버전·hash를 기록했다. 최신이라는 이유로 본문을 읽지 않은 가정을 채택하지 않는다. 논문의 주장과 우리 데이터에서 확인한 결과도 구분한다.

범위 표시: **재독**은 이번에 해당 방법/평가 절을 원문에서 읽었다는 뜻이며 모든 부록 증명의 독립 검증이나 논문 재현 완료를 뜻하지 않는다. **기존+선별**은 보존된 상세 독서 기록을 다시 확인하고 관련 원문 절을 선별 재독했다는 뜻이다. **미확보**는 신규성 판단의 남은 공백이다. 아래 방법의 금융 적용과 실험 설계는 별도 표시하지 않은 경우 우리의 추론/제안이다.

## A. 전체 고객 거래열과 평가

| 원문·버전 / 읽은 범위 | 실제 과제·방법 | 우리 연구와 중복 / 적용 결정 |
|---|---|---|
| [TabularARGN v2](https://arxiv.org/html/2501.12012v2), §3–4/부록 Berka 기존+선별 | 열/시간 순서 autoregression, 정적 parent context, history encoder. 주변·이변량·coherence 중심 평가 | 구현 기반 유지. 고객 조건·과거 거래·관계 보존 자체는 선행. Berka 약 .785 재현과 Sparkov 새 적용을 분리 |
| [CPAR](https://arxiv.org/html/2207.14406v1), §3 재독 | 정적 context와 recurrent history로 다음 행의 분포·종료 확률을 생성 | 충분한 예산의 직접 baseline. 이력/확률적 출력/종료를 추가했다는 주장은 신규 아님 |
| [CPAR 금융 평가](https://arxiv.org/html/2401.00965v1), §3–4 재독 | IBM/Altman 고객 거래, 사기 라벨과 간격. 여러 사기 구성 및 CatBoost/XGBoost/LGBM utility | 사기 라벨을 사용하는 생성 연구는 존재. 자연 비율·모든 조건부 거래 관계를 동시에 입증한 것으로 확대하지 않음 |
| [TabDiT v2, ICLR2025](https://arxiv.org/html/2504.07566v2), §4, §5/공식 저장소 재독 | 행 VAE, 행 내부 AR decoder, 고객 조건부 latent diffusion, EOS로 길이 학습 | 전체 시퀀스 생성·혼합형 관계의 가까운 reference. 공식 공개물이 현재 평가 중심이므로 Sparkov 학습 재현 가능하다고 말하지 않음 |
| [TimeAutoDiff v3, TMLR2025](https://arxiv.org/html/2406.16028v3), §3–5.1/Appendix B/공식 코드 재독 | 혼합형 VAE+latent diffusion, timestamp 조건, 생성·예측·보간·metadata 조건. 카드 데이터는 Sparkov가 아닌 IBM/TabFormer.100명×200거래로 정렬, 기본 window48 | 라벨 포함 혼합형 거래의 직접 비교 후보. 원래 시간 조건부 과제와 시간 자체 생성은 다름. 고객별 가변 길이·1024+ 자유 생성은 별도 적응/평가 필요 |
| [Seq2Synth v3, CIKM2026](https://arxiv.org/html/2607.15606v3), §3–4 재독 | 시간축·횡단·개인 종단·관계구조의 네 fidelity. 13자료/8모델. 이용재 교수 공저 | 새 생성기가 아닌 평가 틀. 기존 전체 점수만의 우위를 버리고 과제에 맞는 timestamp/longitudinal/length 평가를 보강 |
| [TabStruct v2](https://arxiv.org/html/2509.11950v2), §3–5 기존+선별 | 조건부 독립 구조·global utility. static table에서 목표변수 하나 utility가 구조 전체를 대변하지 않음 | 사기탐지기 점수 하나로 생성 품질 판정 금지. 인과 구조 회복은 이 논문의 실험적 근거와 가정 범위를 넘어 우리 결과로 주장하지 않음 |
| [Behavioral fraud benchmark2026](https://arxiv.org/html/2604.13125v1), §3/§7.8/Appendix A 재독 | 시간 간격·burst·공유인프라·velocity 평가. IEEE-CIS/Amazon FDB. 생성 행의 entity assignment 등 평가 설정 포함 | 행동 fidelity 문제 제기 자체는 이미 있음. 이 논문의 ARGN 결과를 우리의 공식 sequential 설정의 구조적 실패 증거로 가져오지 않음. Appendix의 모델 기술·sequence conditioning은 코드 확인이 필요 |
| [PATH2026](https://arxiv.org/html/2602.02766v1), §5/설정·평가 재독 | 사용자 단위 trajectory serialization, DP LoRA LLM, prefix 길이 sampling, 후보 private selection | 고객별 장기 coherence·초기/prefix 학습도 알려짐. 현재 목적에 DP/LLM 전체를 추가하지 않음 |

## B. 시간 출력과 반복 생성: 이번 설계의 가까운 선행

| 원문·버전 / 읽은 범위 | 실제 방법 | 우리에게 주는 결정과 한계 |
|---|---|---|
| [IFL-TPP, ICLR2020](https://arxiv.org/html/1909.12127v2), 혼합 밀도/§5.5 기존+선별 | intensity 없이 conditional lognormal mixture, history/metadata/sequence embedding | 혼합분포·개인 embedding은 새 발명 아님. 일반 history 혼합분포를 반드시 강한 대조로 둠 |
| [Mark-dependent TPP, v2](https://arxiv.org/html/2210.15294v2), §3.2/학습·평가 재독 | mark별 시간분포를 history로 매개변수화해 time–mark 조건부 독립 가정 완화 | 현재 라벨로 간격을 조건화하는 것 자체는 기존. 이 baseline을 빼고 독립 mark/time 모델만 이기면 부족 |
| [MoTPP2026](https://arxiv.org/html/2609.21382v1), §4/§7 재독 | THP backbone+zero-inflated lognormal mixture. 식1은 history 조건의 activity/time 독립. point score에서 항상 최고가 아니지만 calibration/cost 장점 | 가장 최신이라는 이유로 항상 가장 강한 모델은 아님. shared history만으로 충분한 time–mark 결합을 보장하지 않음. 사기 금융 전체열 모델로 오인 금지 |
| [MTDPP, JASA2026](https://doi.org/10.1080/01621459.2026.2657051), [공개 v2 §2](https://arxiv.org/html/2407.03774v2) 재독 | lag별 duration transition mixture. bivariate 성분의 양 marginal을 같게 두어 정상 duration marginal을 구성 | '시간 marginal+의존성 동시 보존'도 이미 연구됨. 강한 통계 대조. 비정상·사기 phase 전환에서 정상성 정리가 자동 적용되지는 않음 |
| [LBDTPP2026 v1](https://arxiv.org/html/2606.24982v1), §IV/설정·공식 코드 기존+선별 | block 간 AR, block 내 latent diffusion, 가변 길이 비동기 사건 생성. local approximation/prefix stability 가정하 이론 | 첫 최신 시간/상태 직접 비교. 실제 마지막 시간을 조건으로 사용하는 평가를 우리의 자유 길이 생성과 그대로 비교하지 않음. block 사용 자체/누적 오류 감소 주장을 신규로 삼지 않음 |
| [SigTPP2026 v1](https://arxiv.org/html/2607.06652v1), §3–4 재독 | interarrival path embedding의 expected signature로 생성 경로 분포를 학습. 비교 가능한 AR architecture | '실제 과거 NLL만 보지 않고 생성 경로 손실'도 알려짐. 우리 새 목적은 개인 clock–간격의 phase/위치별 결합에 추가 효과가 있는지 검증해야 함 |
| [ADiff4TPP2025 v1](https://arxiv.org/html/2504.20411v1), §3 선별 재독 | 사건 VAE+비동기 diffusion/flow schedule+DiT, horizon 조건부 사건 생성 | 시간/mark 공동 생성 및 긴 미래 생성은 기존. 현재 좁힌 주장을 위해 이를 그대로 새 구조에 모두 넣지 않음 |
| [AD-Seq2026 v2](https://arxiv.org/html/2606.06007v2), §2/score matching/§5 재독 | 지정된 정보 흐름을 따르는 과거 조건부 sequential diffusion. ARMA/GP/portfolio 평가 | strict-past/causal 생성 자체는 선행. continuous-series 결과를 mixed fraud sequence 성능으로 대체하지 않음. 미래 없는 정보 흐름 설계의 근거 |
| [Conditional Wasserstein, JMLR2025 v3](https://arxiv.org/html/2403.18705v3), 정의/동기·기존 정리 기록 재독 | 조건을 보존하는 coupling과 조건부 거리의 관계. joint W1만 낮아도 개별 조건이 잘 맞는다고 보장하지 않음 | 구간/phase 손실 및 joint clock–gap 검증의 근거. 조건부 거리나 pooled 상쇄 자체를 우리가 발견한 새 이론이라고 하지 않음 |

## C. 참고하되 이 과제의 해결 보장으로 가져오지 않는 연구

| 원문 / 범위 | 이번 판단 |
|---|---|
| [TabCascade2026 v3](https://arxiv.org/html/2601.22816v3), §3–4 및 기존 실험 독서 재검토 | coarse categorical/numeric 상태와 fine numeric 조건부 출력 분해. 금액 출력 개선의 근거. 고정 시간 구간 보정이나 전체 재귀 안정성을 보장하지 않음 |
| [EmDT2026 v2](https://arxiv.org/html/2603.13566v2), §2.2/§3 재독 | fraud cluster별 diffusion Transformer와 XGBoost 증강 평가. European creditcard 데이터에서 Time 제외. Sparkov 장기 고객열 baseline으로 오인하지 않음 |
| [VAE-GAN+CPAC 최신 공개 v2](https://arxiv.org/html/2507.14706v2), §3.6/§4.5 재독; [KBS2026](https://doi.org/10.1016/j.knosys.2026.116594) | Sparkov 추가 실험이 실제로 있다. 전체 class classifier 신호와 fraud 생성으로 증강/탐지 성능을 평가. 자연 거래열 전체를 생성·관계 보존한 모델은 아님. 'causal' 명칭만으로 인과 보장도 없음. 기존 공개 코드 감사의 class=1/normalization 이슈는 별도 확인 대상 |
| [OmegaSDS2026 v1](https://arxiv.org/html/2605.06315v1), §2–3 가정·구조 재독 | latent dynamics와 switching 연결 자체는 기존. invertible emission·Gaussian dynamics 등 identifiability 가정을 혼합형 금융 데이터에 그대로 적용하지 않음. 이번 시간 head에 큰 SDS를 추가할 근거는 아직 없음 |
| [CTDF2026 v1](https://arxiv.org/html/2606.28674v1), feasibility operator 절 재독 | 규칙 만족을 생성 과정에 넣는 방법. '5초면 사기' 같은 강제 규칙이나 hard constraint 만족은 원하는 확률적 정상/사기 관계 보존과 다름 |
| [F-LLM2026 v2](https://arxiv.org/html/2602.12756v2), §4 재독 | v2는 idealized contraction과 residual 추정 오차 조건을 명시한다. forecasting residual feedback의 안정 조건을 stochastic 거래 생성의 보장으로 쓰지 않음. 이전 실험의 v1 인용을 새 검증 완료처럼 취급하지 않음 |
| [Lin et al.2026 v1](https://arxiv.org/html/2604.27182v1), §III-B/기존 §IV 기록 | 입력 law/조건부 출력 law의 구분만 진단에 참고. 실제 state를 사용하는 보정 알고리즘을 자유 생성에 그대로 쓰거나 논문의 정량 bound를 검증 없이 적용하지 않음 |
| [OMPB, UAI2026](https://proceedings.mlr.press/v337/huang26b.html), 공식 방법 요약 재확인·기존 독서 기록 | frozen forecaster 위 gated calibration, 이후 실제 outcome을 관측하는 predict-then-update. 실제 미래 관측이 없는 우리의 자유 생성은 다른 조건. 보호 조건 사용의 유사성이지 보장 상속이 아님 |

## D. 원문 공백·이전 서베이 연결

첨부 `Diffusion Models in Finance: A Survey`의 tabular 분류와 기존 FinDiff/Imb-FinDiff/DP 계열 독서 기록을 다시 대조했다. 행 합성, minority 증강, 금융 시계열, 고객별 거래열은 관련되지만 같은 과제가 아니다. 이 서베이에 소개된 모든 모델을 현재 거래열 전체 baseline으로 강제하지 않는다.

IFT-GAN(IEEE Access2025, DOI 10.1109/ACCESS.2025.3587793)은 feature selection과 time-aware GAN의 가까운 관련 연구로 유지한다. 이번 IEEE 원문 재접근은 실패했고 기존 독서 기록 이상을 새 전문 정독으로 표시하지 않는다. EntTabDiff와 STG-DGR의 전문 공백도 남아 있다. 이 공백이 있는 상태에서 '동일 연구가 전혀 없다'고 말하지 않는다. 대규모 새 구조 실행 전에 해당 메커니즘과의 중복을 해소해야 하며, 지금은 이미 원문을 확보한 직접 대조를 우선한다.

## E. 공식 코드 확인 — 2026-09-29

| 저장소 | 확인 commit | 코드에서 확인한 제약 |
|---|---|---|
| [TabDiT](https://github.com/fabriziogaruti/TabDiT) | cbb2b9a4866b0071ed9150a552704db188c33d4b | tree17파일, 평가/공개 생성 자료 중심. 새 Sparkov training code 미확인 |
| [TimeAutoDiff](https://github.com/namjoonsuh/TimeAutoDiff) | 1017c9e7010432b6f5200bc15a5b70cfc5d6ec6f | `partition_multi_seq`는 동일 길이, 날짜를 feature에서 빼고 cyclical `time_info`를 별도 제공. `DIFF.sample`도 time_info 요구 |
| [LBDTPP](https://github.com/Zh-Shuai/LBDTPP) | 0ca4817bde2333ce0dcbf156496f7fed9fd48050 | 9월27일 소스 감사와 commit 동일. 첫 event drop·stop_time·test loader 조정 필요 |
| [Seq2Synth](https://github.com/KiwanKwon/Seq2Synth) | 8346a95d6d84876a9ddf9196eb62758082619bd9 | benchmark 평가 reference. 저장소 존재가 8모델 모두 우리의 schema로 재학습 완료를 뜻하지 않음 |

외부 소스는 읽기만 했고 실행하거나 프로젝트에 설치하지 않았다. 기록된 동일 commit에서 필요한 어댑터를 별도 작성하고 공식 원본과의 차이를 공개해야 한다.

## F. 신규성 판단을 위한 정확한 경계

기존 연구가 이미 다룬 것: 고객 이력, mark–time 관계, latent switching, path loss, conditional distance, marginal 보존, prefix/초기 상태, 수치 coarse-to-fine 출력.

이번 후보가 추가로 검증할 것: **실제로 생성한 이력에서 만들어지는 개인 clock과 다음 간격의 결합을 전환/거래 위치별로 제어했을 때, 단순 목적 변경·일반 history head·marginal 보존 모델·최신 block 생성보다 사기 onset 개인 관계와 정상 시간 품질 사이의 비용이 줄어드는가.**

이것은 아직 논문에서 보장된 정답이 아니라, 로컬 진단과 위 선행을 바탕으로 좁힌 검증 가능한 가설이다. 같은 학습 목표를 적용한 강한 대조가 해결한다면 새 architecture 필요성은 기각한다. 이 제한을 명시한 상태에서도 실험과 논문 개발은 진행할 수 있다.
