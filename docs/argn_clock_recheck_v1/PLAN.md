# 개선 여지 재점검 — 2026-09-28 KST

직전 full-length 대조 미채택이 연구 중단이나 최적 성능 증명은 아니다. 사용자 요청에
따라 아래 원인 후보를 제한된 로컬 진단으로 구분한다. 기존 active 모델/HEAD는 유지하며
이 진단을 새 생성기 성능이나 신규 방법 검증으로 부르지 않는다. 최종 test는 읽지 않는다.

1. 표현: a+b exp(-n/50)는200이후 거의 상수다. optimization에서 위치별 독립적인
   scalar offset을 bounded W1 적합해, 요구되는 보정 방향이 해당 함수와 맞는지 본다.
   [1,50),[50,200),[200,500),[500,1024),[1024,2048),[2048,전체)6구간,
   δ∈[-.35,.35], bounded minimize_scalar 최대64평가/구간. 선택은 optimization만.
   internal69고객2noise에서 그대로 평가한다. piecewise oracle 진단이며 배포 모델 아님.
2. 최적화: full-length의 같은두계수/초기0/bounds/목적/seed/tolerance를 유지하고
   maxfev만48→192로 늘려 종료 조건과 개선 여지를 확인한다. 최초48평가를 재현한다.
   더 긴 예산이 자동으로 자유 생성 품질 개선을 뜻하지 않는다.
3. 구성: 저장된 development 생성3팔(GMR/직전prefix/full-prefix)의 동일 고객·위치
   전체지원집합과 real/syn모두2048건 이상인 고정 cohort에서 signed gap/clock 편향,
   W1, 고객동일가중 W1을 함께 본다. 고객을 미리 골라 생성하는 것은 아니며 평가 전용이다.
4. 실제 과거 vs 생성 과거 차이는 기존 조건부/자유 결과와 함께 해석한다. 더 유연한
   출력만으로 해결됐다고 주장하지 않고, 작은 국소 보정과 실제 생성 이력 대조를 분리한다.

가까운 선행과 적용 제한:
- Calinon2016 GMR: 조건부 mixture 자체는 알려진 방법.
- Zheng/Kottas/Sansó JASA2026, Mixture Modeling for Temporal Point Processes with Memory.
  공개 v2(2025-10-29)§2.2 Proposition1,§2.3 Example2를 읽음.
  https://arxiv.org/html/2407.03774v2
  https://doi.org/10.1080/01621459.2026.2657051
  주변분포와 이력 의존을 함께 설계하는 발상은 선행된다. 고정 marginal/초기분포 조건이
  필요한 정리를 개인20-median, 사기전환, 비정상 고객열에 그대로 적용할 수 없다.
- Huang/Ma/Michailidis UAI2026 OMPB §4,§5 ablation:
  https://chenfenghuang.info/assets/OMPB/OMPB.pdf
  frozen backbone+gated residual+source anchor는 알려진 보존 전략이다. delayed real
  outcome feedback이 있는 forecasting이며 synthetic self-feedback에 그 보장을 옮기지 않는다.
- Lin et al.2026 v1§III-B/IV: https://arxiv.org/html/2604.27182v1
  입력분포 이동 관점은 선행된다. 실제 시점값을 쓰는 MCMC를 자유 생성에 복사하지 않는다.
- MeLISA2026 v2§4.2: https://arxiv.org/html/2605.05540v2
  다중 lag increment 제약은 선행된다. 유체 장의 paired reconstruction loss를 금융
  무조건 거래열의 행별 정답 손실로 가져오지 않는다. 평가/진단 관점의 참고다.

같은문제의 최신모델이 이미 실패했다고 주장하지 않는다. 위 접근을 모두 혼합하지 않는다.
다음 실제 모델은 진단 결과를 보고 하나의 변경과 제거 대조로 좁힌다.
