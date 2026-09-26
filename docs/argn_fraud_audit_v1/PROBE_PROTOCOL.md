# 체크포인트 고정 진단

새 생성 결과를 보기 전에 등록한다. 학습/선택/생성은 변경하지 않는다.

- 대상: 과거 A 및 새 공식 ARGN 두 seed의 선택 체크포인트.
- 재사용된 validation 고객147명의 최초512개 거래까지만 사용한다. 원래 전체 길이로 native positional encoding을 계산한 뒤 prefix를 잘라 실제 과거를 보존한다. 최종 test는 읽지 않는다.
- teacher forcing으로 실제 이전 거래와 현재 행의 선행 필드를 제공한다. 열 순서는 native 저장 순서로 고정하고 gap/merchant/amount/category/label의 실제 정답 NLL과 category 정답률을 측정한다. 첫 거래/이후 거래 및 정상/사기를 별도 보고한다. 첫 gap과 padding은 제외한다.
- 현재 merchant의 **column embedding만** 배치 내 다음 고객 값으로 교체한다. history compressor 입력은 그대로 유지한다. 이는 행 안의 merchant 정보 사용 진단이며 실세계 merchant의 인과효과가 아니다.
- 별도 대조에서 첫 위치를 제외한 history compressor 출력을 0으로 만든다. 분포 밖의 제거 진단이므로 이를 최적의 짧은 이력 모델과 동일하게 해석하지 않는다.
- native 손실 계산의 batch 의존성을 동일 batch 복제로 검사한다. 원본 구현을 수정하지 않는다. 각 필드 NLL은 전체 유효 거래 수로 따로 정규화한다.
- 진단 결과만으로 학습 부족이나 특정 경로가 오류의 유일 원인이라고 단정하지 않는다. 체크포인트 hash를 진단 전후 대조한다.
