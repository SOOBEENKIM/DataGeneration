# 검색·선정·독해 범위

검색 기준일은 2026-09-22다. 사용자가 제공한 금융 diffusion 서베이의 tabular 절과 참고문헌을 출발점으로 삼았다. 본문이 인용한 11편을 추적했으며, Table 1의 7개 행만으로 목록을 한정하지 않았다.

추가 검색은 고객조건·혼합 필드·비정규 시간 간격·거래열 생성, 시간/행동/조건부 관계 평가, 장기 생성 오차, 금융 simulator를 기준으로 확장했다. 연도만 새롭고 생성 단위가 다른 논문을 모두 직접 경쟁 연구로 취급하지 않았다. arXiv 버전 이력, 출판사·학회·저자 기관, 공식 저장소를 우선 확인했다. 검색 결과의 AI 요약은 원문 정독의 대체물로 사용하지 않았다.

이번 검색에서 실제 사용한 대표 검색식:

- `"ADiff4TPP" TMLR 2026`
- `"TabCascade" ICML 2026`
- `"LBDTPP" 2026`
- `"2026" "synthetic" "transaction sequences" generation September`
- `"2026" "sequential tabular" generation diffusion August September`
- `"2026" "financial" "synthetic data" "conditional" generation arxiv September`
- `"FINESSE" "Financial Event Sequence" 2026`
- `site:arxiv.org "sequential tabular" "2026" generation September`
- `"Entity-based Financial Tabular Data Synthesis with Diffusion Models" pdf`
- `"EntTabDiff" github`
- `"3698625" pdf`

## 본문 독해에 포함한 집합

서베이 관련 10편 + 추가 11편 = 21편. 제목·원문 URL·버전·페이지·다운로드 checksum은 `sources.json`, 비교용 표는 `paper-matrix.csv`, 실제 내용과 해석은 `reading-notes.md`에 있다. 2025년에 최초 공개되었으나 2026년에 채택/출판/수정된 논문을 최초 공개 2026 논문과 구분해 기록했다.

Seq2Synth는 최초 확보한 v2 대신 최신 v3(2026-08-31)를 다시 확보해 분석했다. FINESSE는 2026-09-09 v1을 읽었다. LBDTPP의 PDF template 머리글은 출판 연도 근거로 사용하지 않았다.

## 추가로 확인했으나 정독 집합에 포함하지 않은 자료

| 자료 | 확인 범위 | 이번 핵심 독해에서의 처리 |
|---|---|---|
| [Diffusion and Flow Matching Models for Tabular Data: A Survey](https://arxiv.org/abs/2502.17119v2) | PDF 금융·관계형 절 및 향후 과제 10–11·17쪽 | 검색 지도로 사용. 소개된 원논문을 모두 읽었다고 주장하지 않음 |
| [Temporal extension of TabDDPM](https://arxiv.org/abs/2604.05257) | 공식 초록·버전 | WISDM sensor window 연구. 주변 분야 후보로 남기며 원문 정독 수에 미포함 |
| [Cluster-based Adaptive Generation](https://doi.org/10.1016/j.frl.2026.110288) | 출판사 초록·2026년 9월 서지 | minority heterogeneity 증강 후보. 원문 미정독, 성능 주장에 사용하지 않음 |
| [Synthetic data in cryptocurrencies](https://arxiv.org/abs/2604.16182) | 공식 초록 | 가격 시계열 중심이라 현재 고객 거래열 비교에서 후순위 |
| [GAN-Diffusion financial time series](https://arxiv.org/abs/2605.27113) | 공식 초록 | 주가·거래량 및 자산 간 상관 중심. 직접 경쟁 집합에서 제외 |
| [ERP Data Provisioning](https://arxiv.org/abs/2607.09712) | 공식 초록 | ERP control testing과 합성자료 내부 검증 중심. 현재 직접 독해 집합에서 제외 |

이 목록은 제외된 모든 검색 결과의 로그가 아니라 관련성이 있어 검토한 주변 후보의 기록이다. healthcare longitudinal generation, relational DB synthesis, 고전 exposure-bias 대책 전체까지 포함한 망라적 신규성 검토는 아직 아니다.

## 접근 제한

EntTabDiff는 DOI, ACM PDF, 공개 대체 경로를 확인했으나 원문을 확보하지 못했다. ResearchGate 역시 full text가 없었다. 2차 서베이의 entity-distribution/cross-attention 설명은 EntTabDiff 원문을 읽었다는 증거로 사용하지 않았다.

ADiff4TPP는 공개 arXiv v1을 읽었다. TMLR 2026 게재는 저자 기관 공개 기록으로 확인했으나 OpenReview 최종본은 접근 확인 화면/HTTP 403 때문에 확보하지 못했다. 원고의 수치를 최종본 수치로 단정하지 않았다.

공개 PDF와 추출 텍스트는 `/tmp/finance-literature-2026/`에 있으며 영구 보관을 보장하지 않는다. 영구 독해 기록에는 재접근 가능한 원문 URL과 파일 checksum을 남겼다. 코드 실행·재학습·독립 reproduction은 수행하지 않았다.
