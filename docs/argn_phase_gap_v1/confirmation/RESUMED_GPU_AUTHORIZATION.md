# 동일 GPU 조건의 추가 draw 실행 — 2026-09-27 후속 요청

사용자가 GPU4장을 확인해 비어 있는 GPU를 골라 실행하도록 명시했고, 다음
시간 출력 비교 실험도 다시 승인했다.4개 GPU가 모두0%/사용자 compute 작업
없음임을 새로 확인했다. 앞서 GPU1 dispatch는 실행되지 않았으며 그 이후 CPU
matched-reference 검증을 완료했다. 당시 거절 및 CPU 결과는 그대로 보존한다.

새 시간 출력은 GPU에서4draw로 비교하므로, phase_mixture와 boundary_fit의
추가 GPU draw21/22도 확보한다. 원래 등록한 GPU1의2작은 worker/8생성,
동일 frozen weight/분포/seed/평가 프로토콜을 그대로 실행한다. 기존 CPU 결과를
GPU 결과와 섞지 않는다. 이 파일은 사용자 후속 권한·필요성 기록이며
원래 protocol 및 manifest hash를 수정하지 않는다. 신규 학습0.
