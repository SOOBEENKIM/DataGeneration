# Sparkov 라벨 포함 ARGN 기준 실험과 원인 대조

사용자가 요청한 **1. 관계 보존 측정, 2. 원인 구분** 단계의 실행 기록이다.
새 CS-SAF 구조나 손실함수의 우수성을 주장하는 실험은 아니다.

- [실험 결과와 해석](RESULTS.md)
- [전체 기준 모델·설정 비교](COMPARISON.md)
- [CPAR 포함 공통 143명 비교](SUPPORTED_COMPARISON.md), [context 지원 제약](CPAR_CONTEXT_SUPPORT_PROTOCOL.md)
- [평가 지표의 의미와 한계](MEASUREMENT_NOTES.md)
- [최초 프로토콜](PROTOCOL.md), [추가 해석 주의점](INTERPRETATION_NOTES.md)
- [Berka 재현 기준](../argn_fraud_audit_v1/BERKA_REPLICATION_RESULTS.md)
- [재실행 순서](REPRODUCE.md)
- [CPAR 실행 완료 기록](CPAR_EXECUTION_RESULTS.md)
- 그림: [147명 관계 비교](figures/relationship_comparison.pdf), [147명 수치 분포](figures/numeric_fidelity.pdf), [143명 공통 관계 비교](figures/supported_relationship_comparison.pdf), [143명 공통 수치 분포](figures/supported_numeric_fidelity.pdf)

최초 AUTO 기준은 보존하고, cap 해제, 고정 가중치 생성 순서 변경, gap만
DIGIT, gap·amount 모두 DIGIT을 별도 실험으로 구분했다. 추가 대조는 해당
문제를 발견한 후 기록한 탐색적 실험이다. 개발 검증 자료를 반복 사용했으며
독립 test 결과로 표현하지 않는다.

대형 학습 모델·생성 parquet·실행 로그는 저장소에서 제외한
`artifacts/sparkov_argn_control_v2/`에 보존한다. 이 문서 폴더에는 집계 지표,
검증 결과, 학습 진행 기록, 소스·가중치·데이터 해시와 PDF/PNG 그림을 남긴다.
실제 고객 ID나 원본 거래 행을 문서에 공개하지 않는다.

최종 전체 출력 검증은 `verification.json`이다. `verification_argn.json`은
CPAR 완료 전에 남긴 ARGN 부분 검증 기록이다. 각 preparation 문서의
`neural_fit_started=false`도 해당 준비 시점의 기록이며 최종 실행 상태는
`fit_summary.csv`와 CPAR 실행 완료 기록에서 확인한다.
