import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | AutoGluon")


@app.cell
def _():
    import os
    import marimo as mo
    import polars as pl
    from datetime import datetime
    from autogluon.tabular import TabularPredictor

    return TabularPredictor, datetime, mo, os, pl


@app.cell
def _(mo):
    mo.md("""
    # Otto: AutoGluon

    `tmp/autogluon-rank-24-in-1-line-of-code.ipynb`(Kaggle)를 marimo로 옮겼습니다.
    `TabularPredictor(...).fit(train, presets="best_quality")` 한 줄로 여러 모델을 학습하고 stacking 합니다.

    - 원본은 `time_limit` 6시간입니다. 여기서는 기본 1시간이고, 환경변수 `AG_TIME_LIMIT`(초)로 바꿉니다.
    - 모델은 `tmp/autogluon/{시각}/`에 저장합니다.
    - 리더보드 1위 모델의 검증 log loss(bagging OOF)를 `eval/scores.csv`에 덧붙이고, test 예측을 `submissions/03_autogluon.csv`에 씁니다.
    """)
    return


@app.cell
def _(os):
    LABEL = "target"
    EVAL_METRIC = "log_loss"
    PRESETS = os.environ.get("AG_PRESETS", "best_quality")
    TIME_LIMIT = int(os.environ.get("AG_TIME_LIMIT", 3600))  # 초 · 원본 6시간
    return EVAL_METRIC, LABEL, PRESETS, TIME_LIMIT


@app.cell
def _(mo, pl):
    project_dir = mo.notebook_dir().parent
    data_dir = project_dir / "data"
    # Enum target 은 문자열로 넘긴다(AutoGluon 이 클래스 이름 Class_1..9 를 그대로 쓴다)
    train = pl.read_parquet(data_dir / "train.parquet").with_columns(pl.col("target").cast(pl.String)).to_pandas()
    test = pl.read_parquet(data_dir / "test.parquet").to_pandas()
    sample = pl.read_parquet(data_dir / "sampleSubmission.parquet")
    mo.md(f"train **{len(train):,}** · test **{len(test):,}**")
    return project_dir, sample, test, train


@app.cell
def _(
    EVAL_METRIC,
    LABEL,
    PRESETS,
    TIME_LIMIT,
    TabularPredictor,
    datetime,
    project_dir,
    train,
):
    run_time = datetime.now()
    model_dir = project_dir / "tmp" / "autogluon" / run_time.strftime("%Y%m%d-%H%M%S")
    predictor = TabularPredictor(
        label=LABEL, eval_metric=EVAL_METRIC, path=str(model_dir),
        learner_kwargs={"ignored_columns": ["id"]},
    ).fit(train, presets=PRESETS, time_limit=TIME_LIMIT)
    return predictor, run_time


@app.cell
def _(pl, predictor):
    leaderboard = pl.from_pandas(predictor.leaderboard(silent=True))
    leaderboard
    return (leaderboard,)


@app.cell
def _(PRESETS, leaderboard, mo, pl, project_dir, run_time):
    # score_val 은 「클수록 좋다」로 뒤집힌 -log_loss 다
    _best = leaderboard.row(0, named=True)
    _row = pl.DataFrame([{
        "time": run_time.strftime("%Y-%m-%d %H:%M:%S"), "notebook": "03_autogluon.py",
        "model": f"ag:{PRESETS}:{_best['model']}", "holdout_logloss": round(-_best["score_val"], 5),
    }])
    _eval_path = project_dir / "eval" / "scores.csv"
    _all = pl.concat([pl.read_csv(_eval_path), _row]) if _eval_path.exists() else _row
    _all.write_csv(_eval_path)
    mo.md(f"최고 모델: **{_best['model']}** · 검증 log loss **{-_best['score_val']:.5f}**")
    return


@app.cell
def _(mo, pl, predictor, project_dir, sample, test):
    with mo.status.spinner("test 예측"):
        _proba = predictor.predict_proba(test)
    _cols = [c for c in sample.columns if c != "id"]  # Class_1..9 순서를 sample 에 맞춘다
    sub = pl.concat([sample.select("id"), pl.from_pandas(_proba[_cols].reset_index(drop=True))], how="horizontal")
    sub_path = project_dir / "submissions" / "03_autogluon.csv"
    sub.write_csv(sub_path)
    print(f"autogluon → {sub_path}")
    mo.md(f"`{sub_path.relative_to(project_dir).as_posix()}` ({sub.height:,} rows)")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
