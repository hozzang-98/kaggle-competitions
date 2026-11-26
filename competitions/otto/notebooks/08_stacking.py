import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | Stacking")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier

    return LGBMClassifier, LogisticRegression, StratifiedKFold, log_loss, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Otto: 직접 스태킹

    `06_oof.py`가 저장한 OOF를 **피처로 써서 2층 모델을 학습**한다.
    AutoGluon이 자동으로 한 것(L1 → L2, 0.48 → 0.42)을 손으로 재현하는 실험.

    | | 입력 | 합치는 방법 |
    |---|---|---|
    | exp02 블렌딩 | 1층 확률 | **고정 가중 평균** |
    | 이 노트북 | 1층 확률 | **2층 모델이 학습** |

    1층 OOF 는 `StratifiedKFold(5, shuffle=True, random_state=24)` 로 만들었다.
    2층도 **같은 분할**을 써야 평가가 정직하다.
    """)
    return


@app.cell
def _(np, pl):
    import pathlib
    project_dir = pathlib.Path.cwd().parent if (pathlib.Path.cwd()/"oof").exists() else pathlib.Path(__file__).resolve().parent.parent
    oof_dir = project_dir / "oof"
    MODELS = ["rf", "xgb", "lgbm", "cb"]
    oofs = {m: np.load(oof_dir / f"oof_{m}.npy") for m in MODELS}
    tests = {m: np.load(oof_dir / f"test_{m}.npy") for m in MODELS}
    train = pl.read_parquet(project_dir / "data" / "train.parquet")
    y = train["target"].to_physical().to_numpy()
    {"oof": {m: v.shape for m, v in oofs.items()}, "y": y.shape}
    return MODELS, oof_dir, oofs, project_dir, tests, y


@app.cell
def _(mo):
    mo.md("""## 2층 입력 만들기 — 4모델 × 9클래스 = 36피처""")
    return


@app.cell
def _(MODELS, np, oofs, tests):
    X_meta = np.hstack([oofs[m] for m in MODELS])       # (61878, 36)
    X_meta_test = np.hstack([tests[m] for m in MODELS])  # (144368, 36)
    {"X_meta": X_meta.shape, "X_meta_test": X_meta_test.shape}
    return X_meta, X_meta_test


@app.cell
def _(mo):
    mo.md("""
    ## 2층 모델 OOF 평가

    1층과 **같은 fold 분할**을 쓴다. 다른 분할을 쓰면 2층이 1층 OOF 를 외워
    CV 가 낙관적으로 나온다.
    """)
    return


@app.cell
def _(
    LGBMClassifier,
    LogisticRegression,
    StratifiedKFold,
    X_meta,
    X_meta_test,
    log_loss,
    np,
    pl,
    y,
):
    LABELS = list(range(9))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
    folds = list(skf.split(X_meta, y))

    META = {
        "lr": lambda: LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        "lgbm_meta": lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
            random_state=24, verbose=-1),
    }

    meta_rows, meta_oof, meta_test = [], {}, {}
    for _name, _make in META.items():
        _oof = np.zeros((len(y), 9))
        _tp = np.zeros((len(X_meta_test), 9))
        for _tr, _va in folds:
            _m = _make().fit(X_meta[_tr], y[_tr])
            _oof[_va] = _m.predict_proba(X_meta[_va])
            _tp += _m.predict_proba(X_meta_test) / len(folds)
        _s = log_loss(y, _oof, labels=LABELS)
        meta_oof[_name], meta_test[_name] = _oof, _tp
        meta_rows.append({"model": _name, "oof_logloss": round(_s, 5)})
        print(f"{_name}: {_s:.5f}", flush=True)
    stack_scores = pl.DataFrame(meta_rows).sort("oof_logloss")
    stack_scores
    return LABELS, meta_oof, meta_test, stack_scores


@app.cell
def _(mo):
    mo.md("""## 비교 — 평균 vs 스태킹""")
    return


@app.cell
def _(LABELS, MODELS, log_loss, np, oofs, pl, stack_scores, y):
    _rows = [{"방법": "lgbm 단독", "oof": round(log_loss(y, oofs["lgbm"], labels=LABELS), 5)}]
    for _combo in [("lgbm", "cb"), ("lgbm", "cb", "xgb"), tuple(MODELS)]:
        _p = np.mean([oofs[m] for m in _combo], axis=0)
        _rows.append({"방법": "평균: " + "+".join(_combo),
                      "oof": round(log_loss(y, _p, labels=LABELS), 5)})
    for _r in stack_scores.iter_rows(named=True):
        _rows.append({"방법": f"스태킹: {_r['model']}", "oof": _r["oof_logloss"]})
    comparison = pl.DataFrame(_rows).sort("oof")
    comparison
    return (comparison,)


@app.cell
def _(mo):
    mo.md("""## 제출 파일 생성""")
    return


@app.cell
def _(meta_test, mo, pl, project_dir, stack_scores):
    _best = stack_scores["model"][0]
    _P = meta_test[_best]
    _P = _P / _P.sum(1, keepdims=True)
    _sample = pl.read_parquet(project_dir / "data" / "sampleSubmission.parquet")
    _cols = [c for c in _sample.columns if c != "id"]
    sub = _sample.select("id").with_columns(pl.DataFrame(_P, schema=_cols))
    sub_path = project_dir / "submissions" / "08_stacking.csv"
    sub.write_csv(sub_path)
    print(f"STACK_SUBMISSION {_best} rows={sub.height}")
    mo.md(f"**{_best}** → `submissions/08_stacking.csv` ({sub.height:,} rows)")
    return


if __name__ == "__main__":
    app.run()
