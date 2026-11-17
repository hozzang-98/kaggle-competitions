"""대조 실험: exp02와 동일한 XGB를 전체 데이터로 1회 학습해 제출.
변수는 '5-fold 평균 → 단일 모델' 하나만 바뀐다."""
import numpy as np, polars as pl
from xgboost import XGBClassifier

import os
D = os.path.dirname(os.path.abspath(__file__))
train = pl.read_csv(f'{D}/data/train.csv', infer_schema_length=None)
test = pl.read_csv(f'{D}/data/test.csv', infer_schema_length=None)
feats = [c for c in train.columns if c != 'Activity']

X = train.select(feats).cast(pl.Float64).to_numpy()
y = train['Activity'].to_numpy()
X_test = test.select(feats).cast(pl.Float64).to_numpy()

# exp02와 완전히 동일한 설정
m = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.5, min_child_weight=1,
    objective='binary:logistic', eval_metric='logloss',
    tree_method='hist', n_jobs=-1, random_state=42,
)
m.fit(X, y)                                    # ← 전체 데이터 1회 학습
p = m.predict_proba(X_test)[:, 1]

sub = pl.DataFrame({'MoleculeId': np.arange(1, len(X_test) + 1),
                    'PredictedProbability': p})
out = f'{D}/submission_xgb_single.csv'
sub.write_csv(out)

samp = pl.read_csv(f'{D}/data/svm_benchmark.csv')
chk = pl.read_csv(out)
assert chk.columns == samp.columns and chk.height == 2501
assert chk['MoleculeId'].to_list() == samp['MoleculeId'].to_list()
assert chk.null_count().to_numpy().sum() == 0 and p.min() >= 0 and p.max() <= 1
print(f'포맷 검증 통과 → {out}')
print(f'확률 min {p.min():.4f} / 평균 {p.mean():.4f} / max {p.max():.4f}')
print('(참고) exp02 5-fold 평균: min 0.0012 / 평균 0.5522 / max 0.9991')
