import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | AutoGluon (stacking, no DyStack)")


@app.cell
def _():
    import os
    import marimo as mo
    import polars as pl
    import pandas as pd
    from datetime import datetime
    from autogluon.tabular import TabularPredictor

    return TabularPredictor, datetime, mo, os, pd, pl


@app.cell
def _(mo):
    mo.md("""
    # Otto: AutoGluon — 스태킹 유지, DyStack 비활성화

    `03_autogluon.py` 원본은 **DyStack 예비 fit → 본 학습 전환 지점에서 segfault**가
    두 번 재현됐다 (torch.jit.script 직후). 원본은 수정하지 않고 이 파일을 따로 둔다.

    바꾼 것은 **fit 인자 두 개뿐**이다.

    ```python
    dynamic_stacking=False   # 크래시가 난 전환 지점 자체를 제거
    num_stack_levels=1       # DyStack 이 정하던 값을 직접 지정 → 스태킹은 유지
    ```

    `best_quality` 프리셋과 bagging 은 그대로 쓴다.
    `AG_TIME_LIMIT`(초) · `AG_FOLD_STRATEGY`(ray 문제 시 `sequential_local`) 로 제어한다.
    """)
    return


@app.cell
def _(os):
    LABEL = "target"
    EVAL_METRIC = "log_loss"
    PRESETS = os.environ.get("AG_PRESETS", "best_quality")
    TIME_LIMIT = int(os.environ.get("AG_TIME_LIMIT", 3600))
    FOLD_STRATEGY = os.environ.get("AG_FOLD_STRATEGY", "auto")
    NUM_STACK_LEVELS = int(os.environ.get("AG_STACK_LEVELS", 1))
    NUM_BAG_FOLDS = int(os.environ.get("AG_BAG_FOLDS", 5))
    # torch 모델 로드 시 segfault 가 재현되어 제외한다 (03 2회 + 07 1회)
    EXCLUDED = [m for m in os.environ.get("AG_EXCLUDE", "FASTAI,NN_TORCH").split(",") if m]
    REFIT_FULL = os.environ.get("AG_REFIT_FULL", "1") == "1"
    {"presets": PRESETS, "time_limit": TIME_LIMIT, "fold_strategy": FOLD_STRATEGY,
     "stack_levels": NUM_STACK_LEVELS, "bag_folds": NUM_BAG_FOLDS,
     "excluded": EXCLUDED, "refit_full": REFIT_FULL}
    return (EVAL_METRIC, EXCLUDED, FOLD_STRATEGY, LABEL, NUM_BAG_FOLDS,
            NUM_STACK_LEVELS, PRESETS, REFIT_FULL, TIME_LIMIT)


@app.cell
def _(mo, pl):
    project_dir = mo.notebook_dir().parent
    data_dir = project_dir / "data"
    train = (pl.read_parquet(data_dir / "train.parquet")
             .with_columns(pl.col("target").cast(pl.String)).to_pandas())
    test = pl.read_parquet(data_dir / "test.parquet").to_pandas()
    sample = pl.read_parquet(data_dir / "sampleSubmission.parquet")
    mo.md(f"train **{len(train):,}** · test **{len(test):,}**")
    return project_dir, sample, test, train


@app.cell
def _(
    EVAL_METRIC,
    EXCLUDED,
    FOLD_STRATEGY,
    LABEL,
    NUM_BAG_FOLDS,
    NUM_STACK_LEVELS,
    PRESETS,
    REFIT_FULL,
    TIME_LIMIT,
    TabularPredictor,
    datetime,
    project_dir,
    test,
    train,
):
    run_time = datetime.now()
    model_dir = project_dir / "tmp" / "autogluon_fix" / run_time.strftime("%Y%m%d-%H%M%S")

    _fit_kwargs = dict(
        presets=PRESETS,
        time_limit=TIME_LIMIT,
        dynamic_stacking=False,              # 크래시 구간 제거
        num_stack_levels=NUM_STACK_LEVELS,   # 스태킹은 유지
        num_bag_folds=NUM_BAG_FOLDS,         # 8 → 5 (모델 수·용량 축소)
        excluded_model_types=EXCLUDED,       # torch 계열 제외
    )
    if FOLD_STRATEGY != "auto":
        _fit_kwargs["ag_args_ensemble"] = {"fold_fitting_strategy": FOLD_STRATEGY}

    predictor = TabularPredictor(
        label=LABEL, eval_metric=EVAL_METRIC, path=str(model_dir),
        learner_kwargs={"ignored_columns": ["id"]},
    ).fit(train, **_fit_kwargs)
    print("FIT_DONE", flush=True)

    # bagging 모델을 전체 데이터 단일 모델로 접는다 → 추론 n_folds 배 빠름
    if REFIT_FULL:
        predictor.refit_full()
        print("REFIT_DONE", predictor.model_best, flush=True)

    # fit() 이후 모델을 디스크에서 재로드하면 segfault 가 재현된다(03×2, 07×3).
    # 같은 셀에서 메모리에 상주시킨 뒤 바로 예측해 재로드를 피한다.
    predictor.persist(models="best")
    print("PERSIST_DONE", flush=True)
    test_proba = predictor.predict_proba(test)
    print("PREDICT_DONE", test_proba.shape, flush=True)
    return predictor, run_time, test_proba


@app.cell
def _(mo):
    mo.md("""## 리더보드 — 학습된 모델 수와 실패 목록을 눈으로 확인""")
    return


@app.cell
def _(pd, pl, predictor):
    leaderboard = pl.from_pandas(predictor.leaderboard(silent=True)) \
        if "silent" in predictor.leaderboard.__doc__ else pl.from_pandas(predictor.leaderboard())
    print(f"LEADERBOARD_ROWS={leaderboard.height}")
    for _r in leaderboard.iter_rows(named=True):
        print(f"  {_r['model']:34} val={-_r['score_val']:.5f}")
    leaderboard
    return (leaderboard,)


@app.cell
def _(NUM_BAG_FOLDS, leaderboard, mo):
    # predictor.info() 는 torch 모델을 다시 로드하다 segfault 가 났다. 리더보드로만 집계한다.
    _n = leaderboard.height
    print(f"FITTED={_n}", flush=True)
    mo.md(f"""
    | 항목 | 값 |
    |---|---|
    | 리더보드 모델 수 | **{_n}** |
    | 최고 모델 | **{leaderboard["model"][0]}** |
    | 검증 log loss | **{-leaderboard["score_val"][0]:.5f}** |

    > `score_val` 은 「클수록 좋다」로 뒤집힌 `-log_loss` 다.
    > 이 값은 **{NUM_BAG_FOLDS}-fold bagging + 스태킹 OOF** 기준이라
    > 우리 5-fold 단일 OOF 와 직접 비교할 수 없다. 비교는 LB 로 한다.
    """)
    return


@app.cell
def _(PRESETS, leaderboard, pl, project_dir, run_time):
    _best = leaderboard.row(0, named=True)
    _row = pl.DataFrame([{
        "time": run_time.strftime("%Y-%m-%d %H:%M:%S"), "notebook": "07_autogluon_stack.py",
        "model": f"ag:{PRESETS}:nodystack:{_best['model']}",
        "holdout_logloss": round(-_best["score_val"], 5),
    }])
    _p = project_dir / "eval" / "scores.csv"
    (pl.concat([pl.read_csv(_p), _row]) if _p.exists() else _row).write_csv(_p)
    _row
    return


@app.cell
def _(mo, pl, project_dir, sample, test_proba):
    _proba = test_proba   # fit 셀에서 이미 예측했다 (재로드 회피)
    _cols = [c for c in sample.columns if c != "id"]   # Class_1..9 순서를 sample 에 맞춘다
    sub = pl.concat(
        [sample.select("id"), pl.from_pandas(_proba[_cols].reset_index(drop=True))],
        how="horizontal",
    )
    sub_path = project_dir / "submissions" / "07_autogluon_stack.csv"
    sub.write_csv(sub_path)
    print(f"SUBMISSION_WRITTEN rows={sub.height}")
    mo.md(f"`{sub_path.relative_to(project_dir).as_posix()}` ({sub.height:,} rows)")
    return


if __name__ == "__main__":
    app.run()
