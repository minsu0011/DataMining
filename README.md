# 직접 구현한 Apriori와 의사결정나무

데이터마이닝 수업에서 연관규칙 탐색과 범주형 분류를 Python으로 구현했습니다. 학습기를 호출하는 대신 후보를 만들고 평가해 줄여 가는 내부 과정을 다룹니다.

## 사용 기술과 데이터

Python 표준 라이브러리를 중심으로 구현했습니다. Apriori는 거래별 항목 목록을, 의사결정나무는 header가 있는 TSV를 받습니다. 분류 학습 파일의 마지막 열이 정답 label입니다.

## 모델 구조

[project1/new.py](project1/new.py)는 단일 항목의 support에서 시작해 더 큰 후보 집합을 만들고 최소 support를 만족하는 집합만 남깁니다. 남은 집합에서 규칙을 나누고 support와 confidence를 계산합니다. 빈발 집합을 다음 단계의 출발점으로 사용해 탐색을 줄입니다.

[project2/new.py](project2/new.py)는 entropy와 split information으로 gain ratio를 계산해 분할 속성을 선택하고 재귀적으로 트리를 만듭니다. 값 종류가 많은 속성을 무조건 선호하지 않도록 information gain을 그대로 쓰는 대신 gain ratio를 사용합니다.

## 개발 과정과 판단

첫 과제에서는 후보 생성·빈도 집계·규칙 출력을 나눴습니다. Support 기준에 따라 후보 수와 연산량이 달라지는 Apriori의 특성을 살펴볼 수 있습니다.

두 번째 과제에서는 트리 구성 뒤 새로운 입력을 내려보내는 예측 경로를 연결했습니다. 학습에서 보지 못한 범주에는 하위 leaf label의 다수결을 사용합니다. 간단하지만 학습 행 수로 가중한 다수결과 다르므로 후속 개선에서는 이 차이와 pruning을 함께 다뤄야 합니다.

## 실행

```bash
python project1/new.py 50 transactions.txt rules.txt
python project2/new.py train.tsv test.tsv predictions.tsv
```

50은 최소 support 비율입니다. 분류 결과에는 테스트 입력의 예측 label이 붙습니다.

## 한계

규칙의 confidence는 인과관계가 아닙니다. 트리도 pruning이나 독립 검증 없이 깊이를 늘리면 일반화를 보장할 수 없습니다. 특정 benchmark의 정확도 우위보다는 알고리즘 구현을 이해하는 실습입니다.

[구현 설명](docs/implementation.md) · [소스 목록](docs/source-index.csv)
