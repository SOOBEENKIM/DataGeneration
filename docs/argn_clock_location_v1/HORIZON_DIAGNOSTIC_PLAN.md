# 적합 범위 밖 거래에서 남는 비용 — 사후 진단 등록

clock_prefix의8개 전체 생성 독립 평가에서 정상 gap은두 부모 모두 GMR보다
개선됐지만 한 부모에서는 history_only보다 나빴다. 위치 계수는 처음1024거래로
학습했고 전체 생성은 그 이후도 포함한다. 비용이 학습 horizon 밖에 몰리는지
분리하기 위한 사후 기술 진단이다. 계수·primary metric·screen은 변경하지 않는다.

같은 실제/생성 고객마다 min(real_length,generated_length) 위치까지만 비교한다.
모든 모델의 고객 길이는 같음을 이미 확인했다. 첫 gap과 개인 median이 정의되지
않은 초반을 제외하고 [6,50),[50,200),[200,500),[500,1024),[1024,2048),[2048,∞)를
사용한다. GMR/history_only/clock_prefix/clock_rollout의 정상 gap과 개인 gap,
clock 분포 W1, 실제/생성 표본 수 및 중앙값을 함께 기록한다.

이는 full marginal 주평가를 대체하거나 causal proof가 되지 않는다. 끝 위치를
생성에 입력하지 않고 평가에서만 matched support를 만든다. 새 학습/생성은 없다.
