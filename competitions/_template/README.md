# <대회명>

- **링크**: https://www.kaggle.com/competitions/<slug>
- **기간**: YYYY-MM-DD ~ YYYY-MM-DD
- **태스크**: (예: 이진 분류 / 회귀 / 시계열 예측)
- **평가 지표**: (예: LogLoss, AUC, RMSLE) — **점수 방향(↑/↓)도 적을 것**
- **결과**: Public ??? / Private ??? (??? / ???팀)

## 데이터

```bash
kaggle competitions download -c <slug> -p data/ && unzip -o 'data/*.zip' -d data/
```

| 파일 | 행 × 열 | 설명 |
|------|---------|------|
| train.csv | | |
| test.csv | | |

**제출 포맷** — 추측하지 말고 sample submission을 직접 열어서 적을 것.
(`sample_submission.csv`가 없고 benchmark 파일이 그 역할인 대회도 있다)

```
컬럼:
행 수:
ID 생성 방식:   (test에 ID 컬럼이 없으면 행 순서가 곧 ID)
```

## 기준점

아무것도 안 한 제출의 점수. 베이스라인이 이보다 나쁘면 파이프라인 버그다.

| 전략 | 점수 |
|---|---|
| (예: all-0.5) | |
| (예: 타깃 평균 일괄) | |
| LB 1위 | |

## 접근법

1.
2.

## 실험 로그

| exp | 모델 | 검증 체계 | CV | Public LB | Private LB | 제출 |
|-----|------|-----------|-----|-----------|------------|------|
| exp01 | baseline | 5-fold OOF | | | | |

> 검증 체계가 다른 점수끼리는 비교할 수 없다. 반드시 같은 체계로 재서 비교할 것.

**CV − LB 갭**:
(갭이 크면 과적합이나 분포 shift를 의심한다. 작으면 CV를 믿고 움직여도 된다)

## 회고

- 잘된 것:
- 아쉬운 것:
- 다음에 쓸 것:
