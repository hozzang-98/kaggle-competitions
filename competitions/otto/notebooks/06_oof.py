import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | StratifiedKFold OOF")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    return (
        CatBoostClassifier,
        LGBMClassifier,
        RandomForestClassifier,
        StratifiedKFold,
        XGBClassifier,
        log_loss,
        mo,
        np,
        pl,
        time,
    )


@app.cell
def _(mo):
    mo.md("""
    # Otto: StratifiedKFold OOF 재측정

    `02_baseline.py`와 **똑같은 4개 모델**을 측정 방식만 바꿔 다시 잰다.
    변수는 하나 — holdout 80/20 → 5-fold OOF.

    | 항목 | 02 (holdout) | 06 (OOF) |
    |---|---|---|
    | 학습 횟수 | 모델당 1회 | 모델당 **5회** |
    | 채점 근거 | 12,376행 | **61,878행** |
    | fold 편차 | 알 수 없음 | **측정됨** |

    **`shuffle=True` 필수** — train이 target으로 정렬돼 있다
    (Class_1 → 행 0~1928, ... Class_9 → 56923~61877).
    섞지 않으면 fold가 연속 블록으로 잘려 CV가 망가진다.
    """)
    return


@app.cell
def _(mo, pl):
    project_dir = mo.notebook_dir().parent
    train = pl.read_parquet(project_dir / "data" / "train.parquet")
    test = pl.read_parquet(project_dir / "data" / "test.parquet")
    X_cols = [c for c in train.columns if c not in ("id", "target")]
    classes = train["target"].dtype.categories.to_list()
    X = train.select(X_cols).to_numpy()
    y = train["target"].to_physical().to_numpy()
    X_test = test.select(X_cols).to_numpy()
    {"X": X.shape, "X_test": X_test.shape, "classes": len(classes)}
    return X, X_test, classes, project_dir, y


@app.cell
def _(
    CatBoostClassifier,
    LGBMClassifier,
    RandomForestClassifier,
    XGBClassifier,
):
    # 02_baseline.py 와 동일한 설정. 변수를 하나만 바꾸기 위해 손대지 않는다.
    MODELS = {
        "rf": lambda: RandomForestClassifier(n_jobs=-1, random_state=24),
        "xgb": lambda: XGBClassifier(n_jobs=-1, learning_rate=0.1, random_state=24),
        "lgbm": lambda: LGBMClassifier(learning_rate=0.1, random_state=24, verbose=-1),
        "cb": lambda: CatBoostClassifier(learning_rate=0.1, random_state=24, verbose=0,
                                         thread_count=-1, allow_writing_files=False),
    }
    N_SPLITS = 5
    SEED = 24
    return MODELS, N_SPLITS, SEED


@app.cell
def _(mo):
    mo.md("""## 분할 검증 — fold마다 클래스 비율이 유지되는가""")
    return


@app.cell
def _(N_SPLITS, SEED, StratifiedKFold, X, np, pl, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    _rows = []
    for _i, (_tr, _va) in enumerate(folds):
        _r = {"fold": _i + 1, "n_train": len(_tr), "n_valid": len(_va)}
        _pc = np.bincount(y[_va], minlength=9) / len(_va)
        for _c in range(9):
            _r[f"C{_c+1}"] = round(float(_pc[_c]) * 100, 2)
        _rows.append(_r)
    _overall = {"fold": 0, "n_train": 0, "n_valid": len(y)}
    _pcall = np.bincount(y, minlength=9) / len(y)
    for _c in range(9):
        _overall[f"C{_c+1}"] = round(float(_pcall[_c]) * 100, 2)
    fold_check = pl.DataFrame([_overall] + _rows)
    fold_check
    return folds, skf


@app.cell
def _(mo):
    mo.md("""
    fold 0 = 전체 분포. 아래 5개 fold가 전부 같아야 한다.

    ## OOF 학습 — 4모델 × 5fold = 20회
    """)
    return


@app.cell
def _(MODELS, X, X_test, folds, log_loss, np, pl, time, y):
    LABELS = list(range(9))
    oof_store, test_store, fold_rows = {}, {}, []

    for _name, _make in MODELS.items():
        _oof = np.zeros((len(y), 9))
        _tp = np.zeros((len(X_test), 9))
        _t0 = time.time()
        for _k, (_tr, _va) in enumerate(folds):
            _m = _make().fit(X[_tr], y[_tr])
            _oof[_va] = _m.predict_proba(X[_va])
            _tp += _m.predict_proba(X_test) / len(folds)
            _fl = log_loss(y[_va], _oof[_va], labels=LABELS)
            fold_rows.append({"model": _name, "fold": _k + 1, "logloss": round(_fl, 5)})
            print(f"{_name} fold{_k+1}: {_fl:.5f}")
        oof_store[_name] = _oof
        test_store[_name] = _tp
        print(f"{_name} done in {time.time()-_t0:.1f}s")

    fold_scores = pl.DataFrame(fold_rows)
    fold_scores
    return LABELS, fold_scores, oof_store, test_store


@app.cell
def _(mo):
    mo.md("""## 결과 — OOF vs holdout""")
    return


@app.cell
def _(LABELS, fold_scores, log_loss, np, oof_store, pl, y):
    HOLDOUT = {"rf": 0.60505, "xgb": 0.54886, "lgbm": 0.49480, "cb": 0.50569}

    _rows = []
    for _name, _oof in oof_store.items():
        _f = fold_scores.filter(pl.col("model") == _name)["logloss"].to_numpy()
        _oof_score = log_loss(y, _oof, labels=LABELS)
        _rows.append({
            "model": _name,
            "oof_logloss": round(_oof_score, 5),
            "fold_mean": round(float(_f.mean()), 5),
            "fold_std": round(float(_f.std()), 5),
            "fold_min": round(float(_f.min()), 5),
            "fold_max": round(float(_f.max()), 5),
            "holdout": HOLDOUT[_name],
            "oof_minus_holdout": round(_oof_score - HOLDOUT[_name], 5),
        })
    summary = pl.DataFrame(_rows).sort("oof_logloss")
    summary
    return HOLDOUT, summary


@app.cell
def _(HOLDOUT, mo, summary):
    _oof_rank = summary["model"].to_list()
    _hold_rank = [m for m, _ in sorted(HOLDOUT.items(), key=lambda kv: kv[1])]
    _changed = _oof_rank != _hold_rank
    mo.md(f"""
    | 순위 | holdout | OOF |
    |---|---|---|
    | 1 | {_hold_rank[0]} | **{_oof_rank[0]}** |
    | 2 | {_hold_rank[1]} | {_oof_rank[1]} |
    | 3 | {_hold_rank[2]} | {_oof_rank[2]} |
    | 4 | {_hold_rank[3]} | {_oof_rank[3]} |

    **순위 변동: {"있음" if _changed else "없음"}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 클래스별 손실 분해 — EDA 예측 검증""")
    return


@app.cell
def _(np, oof_store, pl, summary, y):
    _best = summary["model"][0]
    _oof = np.clip(oof_store[_best], 1e-15, 1 - 1e-15)
    _per_row = -np.log(_oof[np.arange(len(y)), y])
    _cnt = np.bincount(y, minlength=9)
    _rows = []
    for _c in range(9):
        _m = y == _c
        _rows.append({
            "class": f"Class_{_c+1}",
            "n": int(_cnt[_c]),
            "ratio%": round(float(_cnt[_c] / len(y) * 100), 2),
            "mean_loss": round(float(_per_row[_m].mean()), 4),
            "total_contrib": round(float(_per_row[_m].sum() / len(y)), 4),
            "contrib%": round(float(_per_row[_m].sum() / _per_row.sum() * 100), 2),
        })
    class_loss = pl.DataFrame(_rows).sort("total_contrib", descending=True)
    class_loss
    return (class_loss,)


@app.cell
def _(class_loss, mo, summary):
    _b = summary["model"][0]
    _top = class_loss.head(3)
    mo.md(f"""
    `{_b}` 기준. `total_contrib` 합 = OOF LogLoss.
    `mean_loss`는 그 클래스 행 하나당 평균 벌점, `contrib%`는 전체 손실 중 비중.

    손실 상위 3: **{", ".join(_top["class"].to_list())}**
    (합계 {_top["contrib%"].sum():.1f}%)

    > EDA는 Class_2/3/4가 서로 닮아(프로파일 상관 0.79~0.93) 손실이 집중될 것으로 봤다.
    > 위 표와 대조한다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 산출물 저장 — 재학습 없이 제출을 만들 수 있게""")
    return


@app.cell
def _(mo, np, oof_store, project_dir, summary, test_store):
    _d = project_dir / "oof"
    _d.mkdir(exist_ok=True)
    _saved = []
    for _name in oof_store:
        np.save(_d / f"oof_{_name}.npy", oof_store[_name])
        np.save(_d / f"test_{_name}.npy", test_store[_name])
        _saved.append(_name)
    summary.write_csv(project_dir / "eval" / "oof_scores.csv")
    mo.md(f"""
    `oof/` 에 저장: {", ".join(f"`{s}`" for s in _saved)} (각 oof + test 예측)
    `eval/oof_scores.csv` 에 요약 저장.

    test 예측은 **5개 fold 모델의 평균**이다. 이것으로 제출하면
    CV(OOF)와 제출물이 같은 모델들에서 나온다.
    """)
    return


if __name__ == "__main__":
    app.run()
