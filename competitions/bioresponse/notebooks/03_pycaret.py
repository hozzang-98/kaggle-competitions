import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Biological Response | PyCaret options")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from pycaret.classification import ClassificationExperiment
    from sklearn.metrics import log_loss

    return mo, np, pd, LGBMClassifier, ClassificationExperiment, log_loss


@app.cell
def _(mo):
    mo.md("""
    # Biological Response: PyCaret 옵션 비교

    기존 LightGBM 설정을 기준선으로 두고 PyCaret의 `feature_selection`,
    sigmoid 확률 보정, soft blend를 **같은 분할**에서 비교합니다.
    모델 선택 기준은 3-fold CV LogLoss이며, 별도 20% holdout LogLoss도 표시합니다.
    PyCaret 4.0.0a8은 LogLoss 정렬을 거꾸로 처리하므로 직접 정렬합니다.
    """)
    return


@app.cell
def _(mo, pd):
    project_dir = mo.notebook_dir().parent
    train_data = pd.read_csv(project_dir / "data" / "train.csv")
    test_data = pd.read_csv(project_dir / "data" / "test.csv")
    assert test_data.columns.tolist() == train_data.drop(columns="Activity").columns.tolist()
    return project_dir, train_data, test_data


@app.cell
def _(ClassificationExperiment, train_data, log_loss):
    def make_experiment(feature_selection):
        exp = ClassificationExperiment(
            target="Activity", session_id=314, train_size=0.8,
            fold=3, fold_strategy="stratifiedkfold", n_jobs=2,
            preprocess=True, normalize=False,
            feature_selection=feature_selection, verbose=False,
        ).fit(train_data)
        exp.add_metric(
            "logloss", "LogLoss", log_loss,
            target="pred_proba", greater_is_better=False,
        )
        return exp

    baseline_exp = make_experiment(False)
    selected_exp = make_experiment(True)
    assert baseline_exp.X_test.index.equals(selected_exp.X_test.index)
    return baseline_exp, selected_exp


@app.cell
def _(LGBMClassifier):
    lgbm_params = dict(
        random_state=314, n_jobs=2, verbosity=-1,
        subsample_freq=20, subsample=0.9,
        reg_lambda=1.2, reg_alpha=1.1,
        num_leaves=200, n_estimators=700,
        min_split_gain=0.4, max_depth=15,
        colsample_bytree=0.8,
    )
    return lgbm_params


@app.cell
def _(baseline_exp, LGBMClassifier, lgbm_params):
    baseline_result = baseline_exp.create_model(LGBMClassifier(**lgbm_params))
    return baseline_result


@app.cell
def _(selected_exp, LGBMClassifier, lgbm_params):
    selected_lgbm = selected_exp.create_model(LGBMClassifier(**lgbm_params))
    selected_catboost = selected_exp.create_model("catboost")
    selected_rf = selected_exp.create_model("rf")
    return selected_lgbm, selected_catboost, selected_rf


@app.cell
def _(selected_exp, selected_catboost, selected_rf, selected_lgbm):
    calibrated_catboost = selected_exp.calibrate_model(
        selected_catboost.pipeline, method="sigmoid",
    )
    blended = selected_exp.blend_models(
        [selected_catboost.pipeline, selected_rf.pipeline, selected_lgbm.pipeline],
        method="soft",
    )
    return calibrated_catboost, blended


@app.cell
def _(
    pd, np, log_loss, baseline_exp, selected_exp, baseline_result,
    selected_lgbm, selected_catboost, selected_rf, calibrated_catboost, blended,
):
    variants = {
        "baseline_lgbm": (baseline_exp, baseline_result),
        "selected_lgbm": (selected_exp, selected_lgbm),
        "selected_catboost": (selected_exp, selected_catboost),
        "selected_rf": (selected_exp, selected_rf),
        "calibrated_catboost": (selected_exp, calibrated_catboost),
        "soft_blend": (selected_exp, blended),
    }
    rows = []
    for name, (exp, result) in variants.items():
        pipeline = result.pipeline
        class_one = list(pipeline.classes_).index(1)
        prob = pipeline.predict_proba(exp.X_test)[:, class_one]
        rows.append({
            "Variant": name,
            "CV_LogLoss": float(result.metrics.loc["Mean", "LogLoss"]),
            "Holdout_LogLoss": float(log_loss(exp.y_test, prob)),
        })
    comparison = pd.DataFrame(rows).sort_values("CV_LogLoss").reset_index(drop=True)
    best_name = str(comparison.loc[0, "Variant"])
    best_exp, best_result = variants[best_name]
    comparison
    return comparison, best_name, best_exp, best_result


@app.cell
def _(best_exp, best_result, test_data, np, pd):
    final_model = best_exp.finalize_model(best_result.pipeline).pipeline
    _class_one = list(final_model.classes_).index(1)
    probability = final_model.predict_proba(test_data)[:, _class_one]
    submission = pd.DataFrame({
        "MoleculeId": np.arange(1, len(test_data) + 1),
        "PredictedProbability": probability,
    })
    assert submission["PredictedProbability"].between(0, 1).all()
    submission.head(10)
    return submission


@app.cell
def _(project_dir, comparison, best_name, submission, mo):
    output_dir = project_dir / "submissions"
    output_dir.mkdir(exist_ok=True)
    eval_dir = project_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    comparison_path = eval_dir / "pycaret_03_comparison.csv"
    submission_path = output_dir / "pycaret_03_submission.csv"
    comparison.to_csv(comparison_path, index=False)
    submission.to_csv(submission_path, index=False)
    mo.md(
        f"선택: **{best_name}**  \n"
        f"비교 결과: `{comparison_path}`  \n"
        f"제출 파일: `{submission_path}` ({len(submission):,}행)"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    제출 명령:

    ```powershell
    .venv/Scripts/python.exe scripts/kaggle_submit.py notebooks/03_pycaret.py submissions/pycaret_03_submission.csv
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
