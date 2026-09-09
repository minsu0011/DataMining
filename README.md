# Data Mining from First Principles
Python으로 빈발항목 집합/연관규칙과 gain-ratio 의사결정나무를 구현한 실습입니다. 고수준 학습기 호출보다 알고리즘 내부 단계에 초점을 둡니다.

## 두 알고리즘
`project1/new.py`: transaction에서 Apriori 빈발 집합을 구하고 support/confidence를 갖는 규칙을 출력합니다.
`project2/new.py`: entropy와 gain ratio로 속성을 선택하고 재귀적으로 트리를 구축합니다. 알려지지 않은 branch는 하위 leaf label의 다수결로 처리합니다. 이는 훈련 sample 수로 가중한 다수결과 다를 수 있습니다.

## 실행
`python project1/new.py 50 transactions.txt rules.txt`
`python project2/new.py train.tsv test.tsv predictions.tsv`
후자는 header가 있는 tab-separated 파일을 받으며 train의 마지막 열이 label입니다. 데이터 권리는 사용자가 확인해야 합니다.

## 검증
작은 합성 transaction과 분류 예제로 연산을 확인합니다. 원래 benchmark 정확도를 이번에 다시 계산하지 않습니다. 결과를 다른 split/target의 수치와 직접 비교하지 않습니다.

## 한계
Apriori 후보 수가 급증할 수 있고 decision tree에는 일반화 성능을 보증할 pruning/cross-validation 절차가 없습니다. unseen category fallback을 실제 운영 분포에서 검증하지 않았습니다. [검증과 구조](docs/implementation.md)를 참고하세요.

