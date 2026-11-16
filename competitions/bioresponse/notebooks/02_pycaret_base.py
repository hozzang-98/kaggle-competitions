import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Biological Response | PyCaret")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pycaret.classification import ClassificationExperiment
    from sklearn.metrics import log_loss

    return mo, np, pd, ClassificationExperiment, log_loss


@app.cell
def _(mo):
    mo.md("""
    # Biological Response: PyCaret 모델 비교

    PyCaret 4의 분류 실험으로 여러 모델을 3-fold 교차 검증합니다.
    **LogLoss가 가장 낮은 모델**을 전체 훈련 데이터로 다시 학습하고
    `submissions/pycaret_submission.csv`를 생성합니다.
    Kaggle 제출은 별도 스크립트에서 실행합니다.
    """)
    return


@app.cell
def _(mo):
    project_dir = mo.notebook_dir().parent
    data_dir = project_dir / "data"
    return project_dir, data_dir


@app.cell
def _(pd, data_dir):
    train_data = pd.read_csv(data_dir / "train.csv")
    test_data = pd.read_csv(data_dir / "test.csv")
    assert test_data.columns.tolist() == train_data.drop(columns="Activity").columns.tolist()
    train_data.head()
    return train_data, test_data


@app.cell
def _(ClassificationExperiment, log_loss, train_data):
    experiment = ClassificationExperiment(
        target="Activity", session_id=314, train_size=0.8,
        fold=3, fold_strategy="stratifiedkfold", n_jobs=2,
        preprocess=True, normalize=False, verbose=False,
    )
    experiment.fit(train_data)
    experiment.add_metric(
        "logloss", "LogLoss", log_loss,
        target="pred_proba", greater_is_better=False,
    )
    return experiment


@app.cell
def _(experiment, mo):
    candidates = ["lr", "rf", "et", "lightgbm", "xgboost", "catboost", "rbfsvm"]
    comparison = experiment.compare_models(
        include=candidates, sort="AUC", n_select=len(candidates),
        turbo=False, errors="ignore", verbose=False,
    )
    if comparison.leaderboard.empty:
        raise RuntimeError("PyCaret 모델 비교에서 학습에 성공한 모델이 없습니다.")
    # PyCaret 4.0.0a8의 compare_models(sort='LogLoss')는 내림차순으로
    # 정렬하므로 LogLoss 선택은 명시적으로 오름차순 정렬합니다.
    ranking = comparison.leaderboard.sort_values("LogLoss").reset_index(drop=True)
    fitted_models = dict(zip(comparison.ranked_ids, comparison.models))
    best_id = str(ranking.loc[0, "Model"])
    best_model = fitted_models[best_id]
    mo.md(f"최저 교차 검증 LogLoss: **{best_id} / {ranking.loc[0, 'LogLoss']:.5f}**")
    return ranking, best_id, best_model


@app.cell
def _(ranking):
    ranking[["Model", "LogLoss", "AUC", "Accuracy"]]
    return


@app.cell
def _(experiment, best_model, log_loss, mo):
    holdout_prob = best_model.predict_proba(experiment.X_test)
    _positive_class = list(best_model.classes_).index(1)
    holdout_loss = log_loss(experiment.y_test, holdout_prob[:, _positive_class])
    mo.md(f"선택 모델의 보류 검증 LogLoss: **{holdout_loss:.5f}**")
    return holdout_loss


@app.cell
def _(experiment, best_model, test_data, np, pd):
    final_model = experiment.finalize_model(best_model).pipeline
    _positive_class = list(final_model.classes_).index(1)
    probability = final_model.predict_proba(test_data)[:, _positive_class]
    submission = pd.DataFrame({
        "MoleculeId": np.arange(1, len(test_data) + 1),
        "PredictedProbability": probability,
    })
    submission.head(10)
    return submission


@app.cell
def _(project_dir, submission, ranking, best_id, holdout_loss, mo):
    output_dir = project_dir / "submissions"
    output_dir.mkdir(exist_ok=True)
    eval_dir = project_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    submission_path = output_dir / "pycaret_submission.csv"
    comparison_path = eval_dir / "pycaret_comparison.csv"
    submission.to_csv(submission_path, index=False)
    ranking.to_csv(comparison_path, index=False)
    mo.md(
        f"선택 모델: **{best_id}**, 보류 검증 LogLoss: **{holdout_loss:.5f}**  \n"
        f"제출 파일: `{submission_path}` ({len(submission):,}행)  \n"
        f"비교 결과: `{comparison_path}`"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    결과를 확인한 뒤 프로젝트 루트에서 제출합니다.

    ```powershell
    .venv/Scripts/python.exe scripts/kaggle_submit.py notebooks/02_pycaret_base.py submissions/pycaret_submission.csv
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
