"""베이스라인 템플릿.

전처리는 polars, 학습 직전에만 .to_numpy() — sklearn/LightGBM은 polars를 직접 먹지 못한다.
CV를 먼저 확인하고 submission을 만든다. 포맷 검증 assert는 지우지 말 것.
"""
import os

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TARGET = 'target'      # TODO: 실제 타깃 컬럼명
ID = 'id'              # TODO: 제출 ID 컬럼명 (test에 ID가 없으면 행 순서로 생성)
SEED = 42
N_FOLD = 5

# infer_schema_length=None: polars 기본 추론은 앞 100행만 본다.
# 앞부분이 전부 정수인 float 컬럼을 i64로 잡고 ComputeError가 나는 사고가 잦다.
train = pl.read_csv(f'{DATA}/train.csv', infer_schema_length=None)
test = pl.read_csv(f'{DATA}/test.csv', infer_schema_length=None)

feats = [c for c in train.columns if c not in (TARGET, ID)]
# 컬럼 순서를 고정한다 — 트리 모델은 피처를 무작위 샘플링하므로
# 순서가 바뀌면 random_state가 같아도 결과가 달라진다.
X = train.select(feats).cast(pl.Float64).to_numpy()
y = train[TARGET].to_numpy()
X_test = test.select(feats).cast(pl.Float64).to_numpy()

assert X.shape[1] == X_test.shape[1]
assert not np.isnan(X).any() and not np.isnan(X_test).any()
print(f'train {X.shape} / test {X_test.shape}')


def make_model():
    # TODO: 모델 교체
    return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)


skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
test_pred = np.zeros(len(X_test))

for k, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
    m = make_model()
    m.fit(X[tr_i], y[tr_i])
    # TODO: 지표가 LogLoss/AUC면 predict_proba, Accuracy/RMSE면 predict
    oof[va_i] = m.predict_proba(X[va_i])[:, 1]
    test_pred += m.predict_proba(X_test)[:, 1] / N_FOLD
    print(f'  fold {k}  {log_loss(y[va_i], oof[va_i]):.5f}')

cv = log_loss(y, oof)
print(f'\nCV (OOF) : {cv:.5f}')

# ---- submission ----
# test에 ID 컬럼이 없는 대회가 있다. 그때는 행 순서가 곧 ID다 — 절대 섞지 말 것.
sub = pl.DataFrame({
    ID: test[ID] if ID in test.columns else np.arange(1, len(X_test) + 1),
    TARGET: test_pred,
})
out = os.path.join(BASE, 'submissions', 'sub_001.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
sub.write_csv(out)

# ---- 포맷 검증 ----
# sample_submission.csv가 없고 benchmark 파일이 그 역할인 대회도 있다. 직접 열어서 확인할 것.
sample_path = f'{DATA}/sample_submission.csv'
if os.path.exists(sample_path):
    sample = pl.read_csv(sample_path)
    chk = pl.read_csv(out)
    assert chk.columns == sample.columns, f'컬럼 불일치: {chk.columns} vs {sample.columns}'
    assert chk.height == sample.height, f'행 수 불일치: {chk.height} vs {sample.height}'
    assert chk.null_count().to_numpy().sum() == 0, '결측 있음'
    print(f'포맷 검증 통과 → {out}')
else:
    print(f'!! sample_submission.csv 없음 — 포맷을 직접 확인할 것 → {out}')

print(f'제출 메모용 → "exp01 <모델명> cv{cv:.4f}"')
