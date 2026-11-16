# Predicting a Biological Response

2012년 대회. 분자의 화학적 기술자 1776개(`D1`~`D1776`, 전부 익명화)로
생물학적 반응 여부(`Activity`)를 예측한다.

- **평가지표: LogLoss** (낮을수록 좋음). 확률을 제출한다 — `predict_proba`, `scoring='neg_log_loss'`
- **File competition** — CSV 제출. 2026년 현재도 **Late Submission 가능** (확인됨)
- 종료 대회라 제출 즉시 **Public / Private 점수가 둘 다** 보인다

## 데이터

| 파일 | 크기 | 내용 |
|---|---|---|
| `data/train.csv` | 3,751 × 1,777 | `Activity` + D1~D1776 |
| `data/test.csv` | 2,501 × 1,776 | **ID 컬럼 없음 — 행 순서가 곧 MoleculeId** |
| `data/svm_benchmark.csv` | 2,501행 | `sample_submission.csv`가 없어 이게 그 역할 |

제출 포맷: `MoleculeId,PredictedProbability` (헤더 있음, MoleculeId는 1~2501 연번)

## 기준점

| 전략 | LogLoss |
|---|---|
| all-0.5 | 0.69315 |
| train 평균(0.5423) 일괄 | 0.68962 |
| Public LB 1위 | 0.39275 |
| Private(최종) LB 1위 | 0.37355 |

## 실험 로그

| exp | 모델 | 검증 체계 | 검증 점수 | Public LB | Private LB | 제출 |
|---|---|---|---|---|---|---|
| **exp01** | RandomForest (500, sqrt, leaf2) | 5-fold OOF | 0.46423 | **0.46180** | **0.42650** | O |
| exp02 | XGBoost (500, lr0.05, depth6) | 5-fold OOF | 0.47682 | — | — | X |
| nb01 | LightGBM (고정 파라미터) | holdout 20% 1회 | 0.43591 | — | — | X |
| nb02 | CatBoost (PyCaret 7종 비교 1위) | PyCaret 3-fold | 0.47090 | — | — | X |
| nb03 | soft_blend (catboost+rf+lgbm) | PyCaret 3-fold | 0.46130 | — | — | X |

> 검증 체계가 서로 달라 **열끼리 직접 비교할 수 없다.** 같은 자로 재려면 동일한 5-fold OOF에 태워야 한다.

### exp01 — CV와 LB가 거의 일치

CV 0.46423 → Public 0.46180. **갭 −0.0024**로 사실상 동일하다.
Titanic의 CV−LB 갭이 −0.0476이었던 것과 대비된다. train 3,751행 / test 2,501행으로
Titanic(891행)보다 크고, EDA에서 train·test 분포 shift가 없음을 확인한 것과 일치한다.
**이 대회에서는 CV를 믿고 움직여도 된다.**

순위는 371 / 702 (상위 52.8%), Public LB 중앙값 0.45865 바로 아래.

## EDA 요약 (`eda.ipynb`)

- 결측 **0**, 전 피처 **[0, 1]** 정규화 완료 → 전처리 불필요
- 타깃 54.2% / 45.8%
- 상수 0개 · 완전중복 0개 · near-constant 0개 → **제거할 피처가 없다**
- 피처 간 상관 157만 쌍 전수 계산 결과 **|r| > 0.9가 정확히 0개** (최댓값 0.8999)
  → 주최측이 r>0.9 피처를 미리 제거해둔 것으로 보인다
- 타깃 상관은 D27(+0.472) 하나만 압도적, |r|>0.2는 12개뿐
  → 약한 신호가 넓게 분산. 피처 선택보다 전체를 쓰는 트리 앙상블이 맞다
- unique=2인 피처 835개 중 **D823만 `{0, 0.1667}`** — "이진=0/1" 가정은 틀린다

## 구조

```
competitions/bioresponse/
  eda.ipynb / eda.py          EDA (polars 기반)
  EDA/*.png                   차트
  baseline.py                 RF / XGB  — python3 baseline.py --model rf|xgb
  submission_rf.csv           exp01 (제출함)
  submission_xgb.csv          exp02
  notebooks/
    01_lgbm.py                marimo — LGBM
    02_pycaret_base.py        marimo — PyCaret 모델 비교
    03_pycaret.py             marimo — feature_selection / 보정 / blend 비교
    .venv-pc4/                py3.11 + pycaret 4.0.0a8 (02·03 전용)
    out/*.html                실행 결과
  submission.csv                        nb01
  submissions/pycaret_submission.csv    nb02
  submissions/pycaret_03_submission.csv nb03
  eval/*.csv                  PyCaret 비교 결과
```

## 함정 기록

**polars로 wide CSV 읽기 — 기본 스키마 추론이 깨진다**
앞 100행만 보고 `D651`을 `i64`로 잡아 `ComputeError`. → `infer_schema_length=None`
같은 컬럼이 train은 `Int64` / test는 `Float64`로 추론돼 `pl.concat`에서 `SchemaError`. → 양쪽 `.cast(pl.Float64)`

**PyCaret 4.0.0a8은 Python ≥3.11 필요**
02·03이 쓰는 `experiment.fit()`, `comparison.leaderboard`, `result.metrics`는 PyCaret 4 API다.
PyCaret 3.3.2에서는 구조적으로 못 돌린다. → `uv venv --python 3.11`로 별도 환경

**설치물 손상이 엉뚱한 에러로 둔갑한다**
`pillow`의 `top_level.txt`가 깨진 바이너리 → `get_installed_modules()`가 통째 예외
→ 모든 soft dependency 체크 False → LGBM/XGB/CatBoost 컨테이너 제외
→ `Unknown model id 'lightgbm'`. pillow와 무관해 보이는 메시지가 뜬다.
`errors="ignore"` 탓에 **7개 중 4개만 학습된 채로 결과가 나왔고, 1위 모델이 뒤바뀌어 있었다.**
→ `uv pip install --reinstall --no-cache`. 모델 개수를 항상 눈으로 확인할 것.

**AUC 순위와 LogLoss 순위는 다르다**
xgboost는 AUC 0.853(상위권)인데 LogLoss 0.596(최하위권). 순위는 잘 매기지만 확률 보정이 안 된 것.
LogLoss 대회에서는 이 차이가 그대로 손실이 된다.
반대로 catboost는 이미 보정이 좋아서 **sigmoid 보정을 추가하니 오히려 나빠졌다** (0.4700 → 0.4751).

## 다음

- [ ] nb01/nb03을 동일한 5-fold OOF에 태워 exp01과 공정 비교
- [ ] soft_blend 제출해서 CV−LB 갭이 exp01과 같은 패턴인지 확인
- [ ] 상위권(0.39 이하)과의 격차 원인 — 피처 수가 많아 정규화/차원축소 여지
