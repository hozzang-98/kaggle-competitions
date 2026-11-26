# Otto Group Product Classification Challenge

- **링크**: https://www.kaggle.com/competitions/otto-group-product-classification-challenge
- **기간**: 2015-03-17 ~ 2015-05-18 (종료) — **Late Submission 가능** (2026-11-23 확인)
- **태스크**: 9클래스 다중분류 (상품 → Class_1~Class_9)
- **평가 지표**: multi-class LogLoss — **↓ 낮을수록 좋음**
- **결과**: Public 0.48022 / Private 0.48272 (1,175 / 3,509팀, 상위 33.5%)

93개 익명 카운트 피처로 상품 카테고리를 맞춘다. 피처 의미는 비공개라 도메인 지식을 쓸 수 없다.

## 데이터

```bash
kaggle competitions download -c otto-group-product-classification-challenge -p data/raw
unzip -o 'data/raw/*.zip' -d data/raw
```

> `new_competition.sh`는 `data/`에 받지만 `01_make_parquet.py`는 `data/raw/`를 기대한다.

| 파일 | 행 × 열 | 설명 |
|------|---------|------|
| train.csv | 61,878 × 95 | id + feat_1~93 + target |
| test.csv | 144,368 × 94 | id + feat_1~93 |
| sampleSubmission.csv | 144,368 × 10 | id + Class_1~9 |

**제출 포맷** — 파일명이 `sample_submission.csv`가 아니라 **`sampleSubmission.csv`** (camelCase).

```
컬럼: id,Class_1,Class_2,...,Class_9   (행마다 확률 9개, 합 = 1)
행 수: 144,368
ID   : test.csv의 id와 범위·순서 모두 동일 (1~144368 연속)
```

> sample 자체는 전 행이 `Class_1=1, 나머지 0`이다. 복사해서 일부만 채우면
> 남은 0이 `-log(0) = ∞`가 된다.

## 기준점

| 전략 | 점수 |
|---|---|
| 균등 1/9 (`ln 9`) | **2.19722** — 이보다 나쁘면 파이프라인 버그 |
| 클래스 분포 엔트로피 | **1.95041** — 피처를 안 쓴 점수. 실질 기준선 |
| LB 1위 | 0.38055 |
| LB 중앙값 | 0.54166 |

## 데이터 특성 (04_eda.py)

- 결측 0 / 상수 피처 0 / 완전중복 0 / `|r|>0.9` 0쌍 (최대 0.8241) → **전처리 전부 생략**
- 클래스 불균형 **8.4배** (Class_2 26.05% ↔ Class_1 3.12%) → **StratifiedKFold 필수**
- **train이 target으로 정렬돼 있음** (Class_1 → 행 0~1928, ... Class_9 → 56923~61877)
  → `shuffle=True` 없는 분할은 재앙
- 값은 0~398 정수, 셀의 **79.34%가 0** (행별 non-zero 중앙값 18/93)
- `feat_34`는 **Class_5 전용 마커** (eta² 0.64, Class_5만 non-zero 99%·평균 13.5)
- **Class_2/3/4가 서로 닮음** (프로파일 상관 0.79~0.93, 합쳐서 train의 43%) → 손실 집중 예상
- train/test 분포 차이 없음 (평균 상대차 10% 초과 0개) → adversarial validation 불필요

## 접근법

1. 01 parquet 변환 → 02 원본(holdout) → OOF 재측정 → AutoGluon
2. 베이스라인은 튜닝·피처엔지니어링 없이. 이후 **변수를 하나씩만** 바꿔 기여를 잰다

## 실험 로그

| exp | 모델 | 검증 체계 | CV | Public LB | Private LB | 순위 | 제출 |
|-----|------|-----------|-----|-----------|------------|------|------|
| exp01 | lgbm 기본설정, **전체 train 재학습 단일** | holdout 80/20 | 0.49480 | 0.48022 | 0.48272 | 1,175 | ✅ |
| exp02 | lgbm 기본설정, **5-fold 평균** | 5-fold OOF | 0.49180 | 0.47403 | 0.47616 | 1,102 | ✅ |
| **exp03** | **AutoGluon** best_quality, 5-fold bagging + 2층 스태킹 (21모델) | AG bagging+stack OOF | 0.42300 | **0.41182** | **0.41415** | **68** | ✅ |
| exp04 | 직접 스태킹 — 1층 4모델 OOF(36) → lgbm 2층 | 5-fold OOF | 0.46400 | 0.43773 | 0.43937 | 470 | ✅ |
| exp05 | 직접 리스태킹 — OOF(36) + 원본 93피처 → lgbm 2층 | 5-fold OOF | 0.45972 | **0.42453** | **0.42649** | **277** | ✅ |

> 검증 체계가 다른 점수끼리는 비교할 수 없다. 반드시 같은 체계로 재서 비교할 것.
> exp01의 CV(holdout)와 exp02의 CV(OOF)는 같은 표에 있지만 **다른 체계**다. LB만 직접 비교 가능.

### exp01 상세 — 02_baseline.py 원본 (2026-11-23)

4개 모델을 80/20 holdout(`stratify=train_y, random_state=24`)으로 비교.

| 모델 | holdout LogLoss | 노트북 주석의 예전 private |
|---|---|---|
| **lgbm** | **0.49480** 🥇 | 0.50684 (3위) |
| cb | 0.50569 | **0.49374** (1위) |
| xgb | 0.54886 | 0.53526 |
| rf | 0.60505 | 0.59869 |

예전 결과와 **lgbm↔cb 순위가 뒤집혔다.** 차이 0.011은 holdout 1회로 가르기엔 좁다.

제출 모델 설정 — 사실상 전부 기본값:

```python
LGBMClassifier(learning_rate=0.1, random_state=24, verbose=-1)
# n_estimators=100, num_leaves=31, max_depth=-1
# subsample=1.0, colsample_bytree=1.0, reg_alpha=0, reg_lambda=0
# 9클래스 → 100 라운드 × 9 = 900그루
```

제출 명령:

```bash
kaggle competitions submit -c otto-group-product-classification-challenge \
  -f submissions/02_baseline.csv -m "exp01 lgbm holdout0.49480"
```

제출 전 검증 (전부 통과):

| 검증 | 결과 |
|---|---|
| 컬럼명·순서 sample과 일치 | ✅ |
| 행수 144,368 / id 일치 | ✅ |
| 행 합 = 1 | ✅ (min·max 모두 1.000000) |
| 확률 0인 셀 | **0개** (최솟값 1.8e-08) |
| 예측 클래스 분포 ≈ train 분포 | ✅ (열 순서 간접 검증) |

**CV − LB 갭**: holdout 0.49480 − Public 0.48022 = **−0.01458** (LB가 더 좋음)

holdout 점수는 49,502행으로 학습한 모델의 것이고, 제출물은 61,878행으로 **재학습한 다른 모델**이다.
데이터를 25% 더 본 효과로 보인다. 단, 이 갭은 lgbm 하나의 갭이지 대회의 속성이 아니다
(Bioresponse에서 모델마다 −0.0024 ~ −0.0567로 23배 차이났다).

Public−Private 차이는 0.0025로 작다. test가 144,368행이라 채점이 안정적이다.

### exp02 상세 — 06_oof.py, StratifiedKFold OOF (2026-11-23)

**변수 하나만 바꾼 실험.** 모델·설정·피처 전부 exp01과 동일하고 **제출 방식만** 다르다.

```
exp01 제출 ← train 100%로 재학습한 모델 1개
exp02 제출 ← fold 모델 5개의 test 예측 평균
```

#### 4모델 OOF 재측정 — 순위 변동 없음

`StratifiedKFold(n_splits=5, shuffle=True, random_state=24)`

| 순위 | 모델 | OOF | fold std | fold 범위 | holdout | OOF−holdout |
|---|---|---|---|---|---|---|
| 1 | **lgbm** | **0.49180** | 0.00995 | 0.47926~0.50315 | 0.49480 | −0.0030 |
| 2 | cb | 0.50139 | 0.00753 | 0.49188~0.51001 | 0.50569 | −0.0043 |
| 3 | xgb | 0.54310 | 0.00889 | 0.53240~0.55309 | 0.54886 | −0.0058 |
| 4 | rf | 0.61165 | 0.01479 | 0.59723~0.63850 | 0.60505 | +0.0066 |

Bioresponse에서는 측정 방식을 바꾸자 순위가 뒤집혔는데 **여기선 유지됐다.**
train이 61,878행이라 holdout(12,376행)도 꽤 안정적이었던 것으로 보인다
(Bioresponse는 holdout이 750행이었다).

rf만 OOF가 holdout보다 나빠졌고 fold 편차도 가장 크다. 분할에 민감한 모델이다.

#### lgbm vs cb — fold 쌍비교로 확인

모델 각각의 fold 편차(±0.009)가 두 모델 차이(0.0096)와 비슷해 애매해 보였다.
**같은 fold끼리 쌍으로 비교**하면 공통 변동이 상쇄된다.

| fold | lgbm | cb | 차이 |
|---|---|---|---|
| 1 | 0.49429 | 0.50351 | −0.00922 |
| 2 | 0.48110 | 0.49188 | −0.01078 |
| 3 | 0.50120 | 0.50832 | −0.00712 |
| 4 | 0.50315 | 0.51001 | −0.00686 |
| 5 | 0.47926 | 0.49324 | −0.01398 |

**5/5 fold 전부 lgbm 승.** 차이의 표준편차 0.00293 (단독 편차의 1/3).
fold 점수가 출렁이는 건 fold 자체의 난이도 때문이고 두 모델이 같이 움직인다.
→ **lgbm 1위는 운이 아니다.**

#### 클래스별 손실 분해 (lgbm OOF) — EDA 예측 적중

| class | n | 비율 | 평균 벌점 | 기여 | 기여% |
|---|---|---|---|---|---|
| **Class_3** | 8,004 | 12.9% | 0.9233 | 0.1194 | **24.3%** |
| **Class_2** | 16,122 | 26.1% | 0.4321 | 0.1126 | **22.9%** |
| **Class_4** | 2,691 | 4.4% | 1.3244 | 0.0576 | **11.7%** |
| Class_7 | 2,839 | 4.6% | 0.9888 | 0.0454 | 9.2% |
| Class_6 | 14,135 | 22.8% | 0.1832 | 0.0418 | 8.5% |
| Class_1 | 1,929 | 3.1% | 1.3325 | 0.0415 | 8.5% |
| Class_8 | 8,464 | 13.7% | 0.2610 | 0.0357 | 7.3% |
| Class_9 | 4,955 | 8.0% | 0.4080 | 0.0327 | 6.6% |
| **Class_5** | 2,739 | 4.4% | **0.1147** | 0.0051 | **1.0%** |

- **Class_2/3/4가 손실의 58.9%** — EDA의 프로파일 상관(0.79~0.93) 예측대로
- **Class_5는 1.0%** — `feat_34` 마커로 거의 공짜. 여기 노력해도 소용없다
- Class_6은 Class_2와 크기가 비슷한데 기여가 8.5% vs 22.9%. EDA에서 상관 0.019로 완전히 다른 물건이었다
- Class_1/4는 작은데 평균 벌점이 1.33으로 최고 — 소수 클래스라 모델이 자신 없어 한다

→ **개선 여지는 Class_2/3/4 구분에 있다.**

#### 블렌딩 미리보기 (OOF 저장 덕에 재학습 없이)

| 조합 | OOF |
|---|---|
| lgbm 단독 | 0.49180 |
| **lgbm + cb** | **0.48532** |
| lgbm + cb + xgb | 0.49796 |
| lgbm + cb + xgb + rf | 0.50396 |

약한 모델을 동등 가중으로 섞으면 끌어내린다.

#### 제출 결과

```bash
kaggle competitions submit -c otto-group-product-classification-challenge \
  -f submissions/06_lgbm_fold_avg.csv -m "exp02 lgbm 5fold-avg cv0.49180"
```

| | exp01 (단일) | exp02 (fold 평균) | 차이 |
|---|---|---|---|
| Public | 0.48022 | **0.47403** | **−0.00619** |
| Private | 0.48272 | **0.47616** | **−0.00656** |
| 순위 | 1,175 | **1,102** | **+73계단** |

예측 자체는 많이 다르지 않다 — 평균 절대차 0.00702, 예측 클래스가 바뀐 행 3.5%.
**평균 확신도만 0.8071 → 0.8039로 낮아졌고**, 그 차이가 0.0066을 벌었다.
LogLoss가 "자신 있게 틀리는 것"을 벌하는 지표라서다.

fold 모델들은 **데이터를 80%만 봤는데도 이겼다.** 앙상블 효과가 데이터 손실을 상쇄하고 남는다.

Bioresponse와 같은 방향·비슷한 크기:

| | fold 평균 | 단일 재학습 | 차이 |
|---|---|---|---|
| Bioresponse (Private) | 0.37939 | 0.38552 | −0.00613 |
| **Otto (Private)** | **0.47616** | 0.48272 | **−0.00656** |

#### CV−LB 갭이 벌어진 이유

| exp | CV | Public | 갭 |
|---|---|---|---|
| exp01 | 0.49480 (holdout) | 0.48022 | −0.01458 |
| exp02 | 0.49180 (OOF) | 0.47403 | **−0.01777** |

**OOF와 fold 평균 제출도 엄밀히 같은 물건이 아니다.**

```python
oof[va] = m.predict_proba(X[va])             # 각 행을 모델 1개만 예측 (대입)
test_pred += m.predict_proba(X_test) / 5     # 각 행을 모델 5개가 예측 (누적 평균)
```

OOF는 모델 1개의 성능, 제출은 모델 5개의 앙상블. 그래서 LB가 더 좋다.
**OOF는 fold 평균 제출의 보수적 추정치**로 읽어야 한다.

다만 모든 모델에 같은 방향으로 작용하므로 **모델 선택은 왜곡되지 않는다.**
Bioresponse에서 깨진 건 CV(단일)와 제출(앙상블)의 성격이 달랐기 때문인데,
지금은 양쪽 다 같은 fold 모델들에서 나온다.

#### 산출물

```
oof/oof_{rf,xgb,lgbm,cb}.npy    각 4.2MB — OOF 예측 (61878 × 9)
oof/test_{rf,xgb,lgbm,cb}.npy   각 9.9MB — fold 5개 평균 test 예측 (144368 × 9)
eval/oof_scores.csv
```

**재학습 없이 블렌딩·스태킹을 실험할 수 있다.**


### exp03 상세 — AutoGluon (2026-11-26)

**상위 1.9% (68 / 3,509).** exp02 대비 **1,034계단 상승**.

| | 내 최선 (exp02) | AutoGluon | 차이 |
|---|---|---|---|
| Public | 0.47403 | **0.41182** | −0.06221 |
| Private | 0.47616 | **0.41415** | −0.06201 |

내가 exp01 → exp02 에서 번 것이 −0.00656 이었다. **격차는 그 9.5배.**

LB 1위가 0.38055 이므로 **1위와의 차이는 0.031** 뿐이다.
2015년에 수개월 경쟁한 결과와 15분 학습의 거리가 이 정도다.

#### 리더보드 (21모델, 900초, 5-fold bagging + 2층 스태킹, 실패 0건)

| # | 모델 | val log_loss |
|---|---|---|
| 1 | **WeightedEnsemble_L3** | **0.42300** |
| 2 | LightGBMXT_BAG_L2 | 0.42756 |
| 3 | CatBoost_BAG_L2 | 0.42946 |
| 4 | XGBoost_BAG_L2 | 0.43185 |
| 5 | LightGBM_BAG_L2 | 0.43430 |
| 11 | LightGBM_BAG_L1 | 0.47084 |
| 15 | CatBoost_BAG_L1 | 0.47803 |
| 16 | LightGBMXT_BAG_L1 | 0.48369 |
| 21 | CatBoost_r177_BAG_L1 | 1.17887 |

**L1 단독 모델들은 내 lgbm(OOF 0.49180)과 비슷한 수준이다.**
차이는 모델이 좋아서가 아니라 **쌓아서** 났다.

```
LightGBMXT   L1 0.48369 → L2 0.42756   (−0.056)
CatBoost     L1 0.47803 → L2 0.42946   (−0.049)
XGBoost      L1 0.47772 → L2 0.43185   (−0.046)
```

`CatBoost_r177_BAG_L1` 이 1.17887 로 망가졌지만 가중 앙상블이 알아서 배제한다.
실패가 아니라 정상 동작이다.

#### 제출 검증

| 검증 | 결과 |
|---|---|
| 컬럼명·순서 / 행수 / id | ✅ |
| 행 합 = 1 | ✅ |
| 확률 0인 셀 | **0개** (최솟값 4.0e-06) |

exp02 대비 **예측 클래스가 11,117행(7.7%)에서 다르다.** 평균 확신도 0.8039 → 0.8379.
exp01↔exp02 차이가 3.5% 였으니 **스태킹은 두 배 이상 다른 결정**을 내린다.

#### CV−LB 갭이 세 실험 모두 일정하다

| exp | CV | Public | 갭 |
|---|---|---|---|
| exp01 | 0.49480 (holdout) | 0.48022 | −0.01458 |
| exp02 | 0.49180 (5-fold OOF) | 0.47403 | −0.01777 |
| exp03 | 0.42300 (AG bagging+stack) | 0.41182 | −0.01118 |

검증 체계가 서로 다른데도 방향·크기가 비슷하다.
Bioresponse 에서 갭이 모델마다 23배 차이났던 것과 대조적이다.
**test 가 144,368행으로 크고 drift 가 없어서**(EDA 확인) 채점이 안정적인 덕으로 보인다.
Public−Private 차이도 세 제출 모두 0.0023~0.0066 으로 작다.

> 단, `score_val` 은 bagging + 스태킹 OOF 라 우리 5-fold 단일 OOF 와 **같은 체계가 아니다.**
> 위 표에서 CV 열끼리 비교하면 안 된다. LB 만 직접 비교 가능하다.

#### 실행까지 7번 걸렸다 — macOS libomp 충돌

`fit()` 은 매번 성공하는데 **저장된 모델을 다시 로드하는 모든 동작에서 segfault** 가 났다.
`predict_proba()` / `refit_full()` / `predictor.info()` / 본 학습 재진입 전부 해당.

원인은 **macOS 에서 libomp.dylib 중복 로드**였다. 알려진 이슈다.

- [pytorch#191933](https://github.com/pytorch/pytorch/issues/191933) · [LightGBM#6595](https://github.com/lightgbm-org/LightGBM/issues/6595)

```
torch, scikit-learn  → 자체 번들 libomp
lightgbm, xgboost    → homebrew libomp
→ 한 프로세스에 3개 로드, 워커 스레드 상태가 서로 오염됨
```

**해결: `OMP_NUM_THREADS=1`**

```bash
OMP_NUM_THREADS=1 AG_TIME_LIMIT=900 AG_REFIT_FULL=0 MPLBACKEND=Agg \
  ../.venv-ag/bin/marimo export html 07_autogluon_stack.py -o out/07_autogluon_stack.html
```

대가가 있다 — 단일 스레드라 느려진다.

| | 600초 (멀티스레드, 크래시) | 900초 (단일스레드, 성공) |
|---|---|---|
| 최고 val | 0.41685 | 0.42300 |
| bag folds | 8 | 5 |

시간을 1.5배 줬는데 점수는 0.006 나빠졌다.

> `KMP_DUPLICATE_LIB_OK=TRUE` 는 **쓰면 안 된다** ([pytorch#193520](https://github.com/pytorch/pytorch/issues/193520)).
> 데드락·행을 유발한다. 실제로 별도 프로세스 predict 가 CPU 0% 로 2시간 44분 멈췄다.

전체 기록은 [`AUTOGLUON-ISSUE.md`](./AUTOGLUON-ISSUE.md).

#### 받은 코드는 수정하지 않았다

`03_autogluon.py` 는 원본 그대로 두고 `07_autogluon_stack.py` 를 따로 만들었다.
바꾼 것은 fit 인자뿐이다.

```python
dynamic_stacking=False               # DyStack 전환 지점 제거
num_stack_levels=1                   # 스태킹은 유지
num_bag_folds=5                      # 8 → 5
excluded_model_types=['FASTAI','NN_TORCH']
predictor.persist(models="best")     # fit 과 같은 셀에서 즉시 예측
```


### exp04 / exp05 — 직접 스태킹 (2026-11-26)

AutoGluon 이 이긴 이유가 **스태킹**이라는 것이 리더보드에서 드러났다.

```
AutoGluon LightGBM_BAG_L1  0.47084   ← 내 lgbm OOF 0.49180 과 비슷한 수준
LightGBMXT  L1 0.48369 → L2 0.42756  (−0.056)
CatBoost    L1 0.47803 → L2 0.42946  (−0.049)
```

**1층 모델 자체는 내 것과 비슷하다. 차이는 쌓아서 났다.**
그래서 `06_oof.py` 가 저장해둔 OOF 로 직접 스태킹을 해봤다 — 1층 재학습 불필요.

#### 구조

```
exp02 평균      : 최종 = (lgbm + cb) / 2            ← 가중치 0.5 고정
exp04 스태킹    : 1층 확률 36개 → lgbm 2층이 학습    ← 가중치를 상황마다 학습
exp05 리스태킹  : 36개 + 원본 93피처 → lgbm 2층      ← 2층이 상품도 직접 봄
```

2층이 배우는 것: "rf 혼자 다른 말 하면 rf 가 틀리더라",
리스태킹은 추가로 "feat_34 가 높은 상품이면 cb 를 믿어라" 같은 **조건부 규칙**.

#### 같은 체계(5-fold OOF)로 모은 비교

| 방법 | OOF | 직전 대비 |
|---|---|---|
| lgbm 단독 | 0.49180 | — |
| 평균 lgbm+cb | 0.48532 | −0.00648 |
| 스태킹 **lr**(36) | 0.48477 | −0.00055 |
| 스태킹 **lgbm**(36) | 0.46400 | **−0.02077** |
| 리스태킹 lgbm(129) | 0.45972 | −0.00428 |

**로지스틱 회귀 2층은 사실상 "학습된 평균"이라 평균과 거의 같다**(0.48477 vs 0.48532).
선형 결합이라 모델별 가중치 이상을 못 배운다.
**비선형 2층(LightGBM)이 −0.02077 로 평균의 3배를 벌었다.**

`09_restacking.py` 에서 `stack(36)` 을 다시 돌려 **0.46400 이 소수점까지 재현**됐다.
조건이 동일했다는 증거이고, 이게 맞아야 −0.00428 을 믿을 수 있다.

#### OOF 차이가 LB 에서 3배로 커졌다

| | 스태킹 → 리스태킹 | 배율 |
|---|---|---|
| OOF | 0.46400 → 0.45972 (−0.00428) | — |
| Public | 0.43773 → 0.42453 (**−0.01320**) | 3.1배 |
| Private | 0.43937 → 0.42649 (**−0.01288**) | 3.0배 |

방향은 맞았지만 **크기를 과소평가했다.**

#### CV−LB 갭이 더 이상 일정하지 않다

| exp | 갭 (CV − Public) |
|---|---|
| exp01 | +0.01458 |
| exp02 | +0.01777 |
| exp03 AutoGluon | +0.01118 |
| exp04 스태킹 | **+0.02627** |
| exp05 리스태킹 | **+0.03519** |

exp03 까지는 −0.011~−0.018 로 안정적이었는데 **스태킹에서 깨졌다. 쌓을수록 갭이 커진다.**

추정 원인: 2층이 보는 1층 OOF 는 "4/5 데이터로 학습한 모델"의 예측인데,
제출용 test 예측은 **fold 5개 평균**이다. 후자가 더 좋은 입력이라 실제 성능이 OOF 보다 높다.
리스태킹은 2층이 입력을 더 적극 활용하므로 그 효과가 증폭된다.

> **[[Bioresponse-진행기록]] 교훈 2("한 점으로 일반화하지 않는다")가 또 맞았다.**
> exp01~03 에서 "이 대회 갭은 −0.015쯤"이라 결론냈다면 틀렸다.
> 다만 **모델 선택은 왜곡되지 않았다** — CV 순위와 LB 순위가 5개 전부 일치한다.

#### AutoGluon 과의 격차를 80% 좁혔다

```
시작 (exp02 vs exp03)   0.47616 vs 0.41415   −0.06201
현재 (exp05 vs exp03)   0.42649 vs 0.41415   −0.01234
```

남은 구조적 차이:

| | 내 것 (exp05) | AutoGluon (exp03) |
|---|---|---|
| 2층에 원본 피처 | ✅ | ✅ |
| 1층 모델 수 | **4개 (전부 트리)** | **10개** (신경망·ExtraTrees·파라미터 변형) |
| 층 수 | **2층** | **3층** |

1층 다양성이 가장 유력하다. 내 1층은 전부 트리라 **비슷한 실수를 한다.**


## 환경

```bash
~/.local/bin/uv venv --python 3.11 .venv
~/.local/bin/uv pip install --python .venv/bin/python \
  marimo polars pyarrow pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

- 실행: `MPLBACKEND=Agg ../.venv/bin/marimo export html <nb>.py -o out/<nb>.html`
- AutoGluon은 버전 고정 충돌을 피해 **별도 venv**(`.venv-ag`) 예정

### 걸렸던 것

- **lightgbm `VERSION.txt`가 깨진 바이너리로 설치** → `import lightgbm`이 UnicodeDecodeError.
  `uv pip install --reinstall --no-cache lightgbm`으로 해결.
  Bioresponse의 pillow `top_level.txt` 손상과 같은 유형. 안 잡았으면 02가 3개만 학습하고 정상 종료했을 것
- `seaborn` 미설치 / `eval/` 디렉토리 없음 → 02가 죽을 뻔함. 둘 다 환경에서 해결(코드 미수정)

## 회고

- 잘된 것:
- 아쉬운 것:
- 다음에 쓸 것:
