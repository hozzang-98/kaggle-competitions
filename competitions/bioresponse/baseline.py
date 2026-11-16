"""Bioresponse 베이스라인 — RF / XGBoost.

평가지표는 LogLoss (낮을수록 좋음). predict_proba 로 확률을 제출한다.
EDA 결과: 결측 0, 전 피처 [0,1], 상수/중복/near-constant 없음
→ 피처 제거도 엔지니어링도 하지 않고 1776개 전부 사용.

사용법: python3 baseline.py --model rf|xgb
"""
import argparse
import os
import time

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
SEED = 42
N_FOLD = 5

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['rf', 'xgb'], required=True)
parser.add_argument('--out', default=None, help='submission 파일명')
args = parser.parse_args()

# ---- 로딩 (polars) ----
# infer_schema_length=None: 기본 추론은 앞 100행만 봐서 D651을 i64로 잘못 잡는다
train = pl.read_csv(f'{DATA}/train.csv', infer_schema_length=None)
test = pl.read_csv(f'{DATA}/test.csv', infer_schema_length=None)
feats = [c for c in train.columns if c != 'Activity']

# 컬럼 순서 고정 — 트리 모델은 피처 샘플링을 하므로 순서가 바뀌면 결과가 달라진다
# sklearn/xgboost는 polars를 직접 먹지 못해 학습 직전 numpy 변환
X = train.select(feats).cast(pl.Float64).to_numpy()
y = train['Activity'].to_numpy()
X_test = test.select(feats).cast(pl.Float64).to_numpy()

assert X.shape[1] == X_test.shape[1] == 1776
assert not np.isnan(X).any() and not np.isnan(X_test).any()
print(f'train {X.shape} / test {X_test.shape} / 양성비율 {y.mean():.4f}')


def make_model():
    if args.model == 'rf':
        return RandomForestClassifier(
            n_estimators=500,
            max_features='sqrt',
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        )
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=1,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1,
        random_state=SEED,
    )


# ---- CV ----
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
test_pred = np.zeros(len(X_test))
scores = []

t0 = time.time()
for k, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
    m = make_model()
    m.fit(X[tr_i], y[tr_i])
    oof[va_i] = m.predict_proba(X[va_i])[:, 1]
    test_pred += m.predict_proba(X_test)[:, 1] / N_FOLD
    s = log_loss(y[va_i], oof[va_i])
    scores.append(s)
    print(f'  fold {k}  LogLoss {s:.5f}  ({time.time()-t0:.0f}s)')

cv = log_loss(y, oof)
print(f'\n[{args.model}] CV LogLoss (OOF 전체) : {cv:.5f}')
print(f'[{args.model}] fold 평균 {np.mean(scores):.5f} (±{np.std(scores):.5f})')
print(f'기준점: all-0.5 = 0.69315 / train평균 = 0.68962 / 2012 LB 1위 = 0.37355')
if cv >= 0.68962:
    print('!! 기준점보다 나쁘다 — 파이프라인 버그 의심')

# ---- submission ----
# test.csv에는 ID 컬럼이 없다. 행 순서 = MoleculeId (1..2501)
out = args.out or f'submission_{args.model}.csv'
sub = pl.DataFrame({
    'MoleculeId': np.arange(1, len(X_test) + 1),
    'PredictedProbability': test_pred,
})
sub.write_csv(os.path.join(BASE, out))

# ---- 포맷 검증 (svm_benchmark.csv 가 sample submission 역할) ----
sample = pl.read_csv(f'{DATA}/svm_benchmark.csv')
chk = pl.read_csv(os.path.join(BASE, out))
assert chk.columns == sample.columns, f'컬럼 불일치: {chk.columns} vs {sample.columns}'
assert chk.height == sample.height == 2501, f'행 수 불일치: {chk.height}'
assert chk['MoleculeId'].to_list() == sample['MoleculeId'].to_list(), 'MoleculeId 불일치'
p = chk['PredictedProbability'].to_numpy()
assert chk.null_count().to_numpy().sum() == 0, '결측 있음'
assert p.min() >= 0 and p.max() <= 1, f'확률 범위 벗어남: [{p.min()}, {p.max()}]'
print(f'\n포맷 검증 통과 → {out}')
print(f'  {chk.height}행, 컬럼 {chk.columns}')
print(f'  예측 확률: min {p.min():.4f} / 평균 {p.mean():.4f} / max {p.max():.4f}')
print(f'  제출 메모용 → "exp{"01 rf" if args.model=="rf" else "02 xgb"} cv{cv:.4f}"')
