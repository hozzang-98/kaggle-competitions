import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | Restacking")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import pathlib
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss
    from lightgbm import LGBMClassifier

    return LGBMClassifier, StratifiedKFold, log_loss, mo, np, pathlib, pl


@app.cell
def _(mo):
    mo.md("""
    # Otto: restacking — 2층에 원본 피처를 더한다

    `08_stacking.py`에서 **변수 하나만** 바꾼다.

    | | 2층 입력 |
    |---|---|
    | 08 스태킹 | 1층 확률 36개 |
    | **09 restacking** | 1층 확률 36개 **+ 원본 93피처 = 129개** |

    2층 모델·fold 분할·1층 OOF 는 전부 동일하다.
    AutoGluon 의 L2 가 이 구조라, 그 차이가 얼마인지 재는 것이 목적이다.

    기대하는 것: 2층이 "이 상품이 어떤 상품인가"까지 보고
    `feat_34 가 높으면 cb 를 믿어라` 같은 조건부 규칙을 배우는 것.
    """)
    return


@app.cell
def _(np, pathlib, pl):
    project_dir = pathlib.Path(__file__).resolve().parent.parent
    MODELS = ["rf", "xgb", "lgbm", "cb"]
    oofs = {m: np.load(project_dir / "oof" / f"oof_{m}.npy") for m in MODELS}
    tests = {m: np.load(project_dir / "oof" / f"test_{m}.npy") for m in MODELS}

    train = pl.read_parquet(project_dir / "data" / "train.parquet")
    test = pl.read_parquet(project_dir / "data" / "test.parquet")
    feats = [c for c in train.columns if c.startswith("feat_")]
    X_raw = train.select(feats).to_numpy().astype(np.float32)
    X_raw_test = test.select(feats).to_numpy().astype(np.float32)
    y = train["target"].to_physical().to_numpy()

    X_stack = np.hstack([oofs[m] for m in MODELS])                       # 36
    X_restack = np.hstack([X_stack, X_raw])                              # 36 + 93
    X_stack_test = np.hstack([tests[m] for m in MODELS])
    X_restack_test = np.hstack([X_stack_test, X_raw_test])
    {"stack": X_stack.shape, "restack": X_restack.shape}
    return (
        X_restack,
        X_restack_test,
        X_stack,
        X_stack_test,
        project_dir,
        y,
    )


@app.cell
def _(mo):
    mo.md("""## 두 입력을 같은 조건으로 비교""")
    return


@app.cell
def _(
    LGBMClassifier,
    StratifiedKFold,
    X_restack,
    X_restack_test,
    X_stack,
    X_stack_test,
    log_loss,
    np,
    pl,
    y,
):
    LABELS = list(range(9))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=24).split(X_stack, y))

    def make_meta():
        # 08_stacking.py 의 lgbm_meta 와 동일한 설정
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
            random_state=24, verbose=-1)

    VARIANTS = {
        "stack(36)": (X_stack, X_stack_test),
        "restack(129)": (X_restack, X_restack_test),
    }
    rows, preds = [], {}
    for _name, (_X, _Xt) in VARIANTS.items():
        _oof = np.zeros((len(y), 9))
        _tp = np.zeros((len(_Xt), 9))
        for _tr, _va in folds:
            _m = make_meta().fit(_X[_tr], y[_tr])
            _oof[_va] = _m.predict_proba(_X[_va])
            _tp += _m.predict_proba(_Xt) / len(folds)
        _s = log_loss(y, _oof, labels=LABELS)
        preds[_name] = _tp
        rows.append({"input": _name, "n_features": _X.shape[1], "oof_logloss": round(_s, 5)})
        print(f"{_name}: {_s:.5f}", flush=True)
    result = pl.DataFrame(rows).sort("oof_logloss")
    result
    return preds, result


@app.cell
def _(mo, result):
    _best = result.row(0, named=True)
    _worst = result.row(-1, named=True)
    mo.md(f"""
    | 입력 | 피처 수 | OOF |
    |---|---|---|
    | {_worst["input"]} | {_worst["n_features"]} | {_worst["oof_logloss"]} |
    | **{_best["input"]}** | {_best["n_features"]} | **{_best["oof_logloss"]}** |

    차이 **{_best["oof_logloss"] - _worst["oof_logloss"]:+.5f}**

    참고 — 같은 체계(5-fold OOF)로 잰 값들:

    | 방법 | OOF |
    |---|---|
    | lgbm 단독 | 0.49180 |
    | 평균 lgbm+cb | 0.48532 |
    | 스태킹 lr | 0.48477 |
    | 스태킹 lgbm (36) | 0.46400 |
    """)
    return


@app.cell
def _(mo, pl, preds, project_dir, result):
    _best = result["input"][0]
    _P = preds[_best]
    _P = _P / _P.sum(1, keepdims=True)
    _sample = pl.read_parquet(project_dir / "data" / "sampleSubmission.parquet")
    _cols = [c for c in _sample.columns if c != "id"]
    sub = _sample.select("id").with_columns(pl.DataFrame(_P, schema=_cols))
    sub_path = project_dir / "submissions" / "09_restacking.csv"
    sub.write_csv(sub_path)
    print(f"RESTACK_SUBMISSION {_best} rows={sub.height}", flush=True)
    mo.md(f"**{_best}** → `submissions/09_restacking.csv` ({sub.height:,} rows)")
    return


if __name__ == "__main__":
    app.run()
