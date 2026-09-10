# 후보를 줄이고 분할을 선택하기

Apriori는 단일 항목의 support를 세고 빈발 집합에서 다음 후보를 만듭니다. 최소 support로 후보를 줄인 뒤 연관규칙과 confidence를 계산합니다.

의사결정나무는 label entropy와 split information으로 gain ratio를 계산해 분할 속성을 고릅니다. 보지 못한 범주는 하위 leaf label의 다수결로 처리하며 학습 행 수로 가중한 다수결과 다릅니다.

합성 테스트는 간단한 support·분류 동작을 확인합니다. 실제 benchmark 정확도나 과적합 해소의 증거는 아닙니다.
