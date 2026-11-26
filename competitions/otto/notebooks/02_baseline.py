import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | RF & GBDT baselines")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    return (
        CatBoostClassifier,
        LGBMClassifier,
        RandomForestClassifier,
        XGBClassifier,
        datetime,
        log_loss,
        mo,
        pl,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # Otto: RF · XGB · LGBM · CatBoost 기준선

    `tmp/02 Train - RF.ipynb`(Colab)를 marimo로 옮겼습니다.
    `01_make_parquet.py`가 만든 `data/*.parquet`를 읽습니다.
    4개 모델의 80/20 holdout log loss를 `eval/scores.csv`에 덧붙이고,
    log loss가 가장 낮은 모델을 전체 train으로 다시 학습해 `submissions/02_baseline.csv`를 씁니다.

    예전 Kaggle 점수(private): RF 0.59869 · XGB 0.53526 · LGBM 0.50684 · CatBoost 0.49374
    """)
    return


@app.cell
def _(mo, pl):
    project_dir = mo.notebook_dir().parent
    data_dir = project_dir / "data"
    train = pl.read_parquet(data_dir / "train.parquet")
    test = pl.read_parquet(data_dir / "test.parquet")
    sample = pl.read_parquet(data_dir / "sampleSubmission.parquet")
    mo.vstack([train.head(), test.head()])
    return project_dir, sample, test, train


@app.cell
def _(mo, test, train):
    mo.md(f"""
    train size: **{train.height:,}** · test size: **{test.height:,}** ·
    null: train {train.null_count().sum_horizontal()[0]} / test {test.null_count().sum_horizontal()[0]}
    """)
    return


@app.cell
def _(plt, sns, train):
    _counts = train["target"].value_counts().sort("target")
    _fig, _ax = plt.subplots(1, 2, figsize=(15, 6))
    _ax[0].pie(_counts["count"], labels=_counts["target"], autopct="%1.2f%%",
               explode=[0.1] * _counts.height, shadow=True)
    _ax[0].set_title("target")
    sns.barplot(x=_counts["target"].cast(str).to_list(), y=_counts["count"].to_list(), ax=_ax[1])
    _ax[1].set_title("target")
    _fig
    return


@app.cell
def _(test, train):
    X_cols = [c for c in train.columns if c not in ("id", "target")]
    classes = train["target"].dtype.categories.to_list()
    train_X = train.select(X_cols).to_numpy()
    train_y = train["target"].to_physical().to_numpy()
    test_X = test.select(X_cols).to_numpy()
    return classes, test_X, train_X, train_y


@app.cell
def _(
    CatBoostClassifier,
    LGBMClassifier,
    RandomForestClassifier,
    XGBClassifier,
):
    MODELS = {
        "rf": lambda: RandomForestClassifier(n_jobs=-1, random_state=24),
        "xgb": lambda: XGBClassifier(n_jobs=-1, learning_rate=0.1, random_state=24),
        "lgbm": lambda: LGBMClassifier(learning_rate=0.1, random_state=24, verbose=-1),
        "cb": lambda: CatBoostClassifier(learning_rate=0.1, random_state=24, verbose=0,
                                         thread_count=-1, allow_writing_files=False),
    }
    return (MODELS,)


@app.cell
def _(
    MODELS,
    classes,
    datetime,
    log_loss,
    mo,
    pl,
    project_dir,
    train_X,
    train_test_split,
    train_y,
):
    _X_tr, _X_va, _y_tr, _y_va = train_test_split(
        train_X, train_y, test_size=0.2, stratify=train_y, random_state=24)
    _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _rows = []
    for _name in mo.status.progress_bar(list(MODELS), title="holdout 학습"):
        _m = MODELS[_name]().fit(_X_tr, _y_tr)
        _loss = log_loss(_y_va, _m.predict_proba(_X_va), labels=list(range(len(classes))))
        print(f"{_name}: holdout log loss {_loss:.5f}")
        _rows.append({"time": _now, "notebook": "02_baseline.py", "model": _name,
                      "holdout_logloss": round(_loss, 5)})
    holdout = pl.DataFrame(_rows).sort("holdout_logloss")
    best = holdout["model"][0]  # log loss 는 낮을수록 좋다

    _eval_path = project_dir / "eval" / "scores.csv"
    _all = pl.concat([pl.read_csv(_eval_path), holdout]) if _eval_path.exists() else holdout
    _all.write_csv(_eval_path)
    mo.vstack([mo.md(f"최고 모델: **{best}**"), holdout])
    return (best,)


@app.cell
def _(
    MODELS,
    best,
    classes,
    mo,
    pl,
    project_dir,
    sample,
    test_X,
    train_X,
    train_y,
):
    with mo.status.spinner(f"{best}: 전체 train 학습 · test 예측"):
        _full = MODELS[best]().fit(train_X, train_y)
        _pred = _full.predict_proba(test_X)

    sub = sample.select("id").with_columns(
        pl.DataFrame(_pred, schema=[f"Class_{i + 1}" for i in range(len(classes))]))
    sub_path = project_dir / "submissions" / "02_baseline.csv"
    sub.write_csv(sub_path)
    print(f"{best} → {sub_path}")
    mo.md(f"**{best}** → `{sub_path.relative_to(project_dir).as_posix()}` ({sub.height:,} rows)")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
