import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium", app_title="Biological Response | LightGBM")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from lightgbm import LGBMClassifier
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import ConfusionMatrixDisplay, log_loss
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import MinMaxScaler
    return (
        ConfusionMatrixDisplay,
        LGBMClassifier,
        MinMaxScaler,
        PCA,
        TSNE,
        cross_val_score,
        log_loss,
        mo,
        np,
        pl,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # Biological Response: LightGBM

    원본 노트북의 데이터 탐색, 차원 축소, covariate shift 확인, 검증,
    feature importance, 제출 예측을 marimo 셀로 옮겼습니다.
    계산량이 큰 탐색 단계는 아래  체크박스를 켜면 실행됩니다.
    제출 파일은 프로젝트 루트의 `submission.csv`입니다.
    """)
    return


@app.cell
def _(mo):
    data_dir = mo.notebook_dir().parent / "data"
    project_dir = data_dir.parent
    data_dir
    return data_dir, project_dir


@app.cell
def _(data_dir, pl):
    train_data = pl.read_csv(data_dir / "train.csv", infer_schema_length=None)
    test_data = pl.read_csv(data_dir / "test.csv", infer_schema_length=None)
    train_data.head(10)
    return test_data, train_data


@app.cell
def _(mo, test_data, train_data):
    mo.md(f"""
    훈련: **{train_data.height:,}행 × {train_data.width:,}열**, "
        f"테스트: **{test_data.height:,}행 × {test_data.width:,}열**  \n"
        f"결측값: **{sum(train_data.null_count().row(0)):,}개**  \n"
        f"정답 분포: **{train_data.group_by('Activity').len().sort('Activity').to_dicts()}**
    """)
    return


@app.cell
def _(mo):
    explore = mo.ui.checkbox(label="PCA와 t-SNE 탐색 실행", value=False)
    shift_check = mo.ui.checkbox(label="열별 covariate shift 점검 실행", value=False)
    mo.vstack([explore, shift_check])
    return explore, shift_check


@app.cell
def _(MinMaxScaler, PCA, explore, mo, np, plt, train_data):
    mo.stop(not explore.value)
    exploratory_X = train_data.drop("Activity").to_numpy()
    exploratory_y = train_data["Activity"].to_numpy()
    exploratory_scaled = MinMaxScaler().fit_transform(exploratory_X)
    exploratory_pca = PCA().fit(exploratory_scaled)
    fig_variance, ax_variance = plt.subplots()
    ax_variance.plot(np.cumsum(exploratory_pca.explained_variance_ratio_))
    ax_variance.set(xlabel="Components", ylabel="Cumulative explained variance")
    fig_variance
    return exploratory_scaled, exploratory_y


@app.cell
def _(PCA, TSNE, exploratory_scaled, exploratory_y, explore, mo, plt, sns):
    mo.stop(not explore.value)
    # 원본 노트북과 같이 750개 주성분을 거쳐 2차원으로 줄입니다.
    reduced = PCA(n_components=750, random_state=314).fit_transform(exploratory_scaled)
    embedded = TSNE(
        n_components=2, perplexity=50, learning_rate=200,
        max_iter=1000, random_state=314,
    ).fit_transform(reduced)
    fig_tsne, ax_tsne = plt.subplots()
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=exploratory_y, ax=ax_tsne)
    ax_tsne.set(title="PCA + t-SNE")
    fig_tsne
    return


@app.cell
def _(
    LGBMClassifier,
    cross_val_score,
    mo,
    np,
    pl,
    shift_check,
    test_data,
    train_data,
):
    mo.stop(not shift_check.value)
    combined = pl.concat(
        [train_data.drop("Activity"), test_data], how="vertical"
    )
    origin = np.r_[np.zeros(train_data.height), np.ones(test_data.height)]
    shifted_columns = []
    for column in combined.columns:
        scores = cross_val_score(
            LGBMClassifier(n_estimators=50, verbosity=-1, n_jobs=2),
            combined.select(column).to_numpy(), origin,
            cv=2, scoring="roc_auc",
        )
        if scores.mean() > 0.7:
            shifted_columns.append((column, float(scores.mean())))
    mo.md(f"ROC AUC > 0.7인 열: **{shifted_columns or '없음'}**")
    return


@app.cell
def _(MinMaxScaler, train_data, train_test_split):
    features = train_data.drop("Activity").columns
    X = train_data.select(features).to_numpy()
    y = train_data["Activity"].to_numpy()
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=32
    )
    validation_scaler = MinMaxScaler()
    X_fit = validation_scaler.fit_transform(X_fit)
    X_valid = validation_scaler.transform(X_valid)
    return X, X_fit, X_valid, features, y, y_fit, y_valid


@app.cell
def _(LGBMClassifier):
    model_params = dict(
        random_state=314, n_jobs=2, verbosity=-1,
        subsample_freq=20, subsample=0.9,
        reg_lambda=1.2, reg_alpha=1.1,
        num_leaves=200, n_estimators=700,
        min_split_gain=0.4, max_depth=15,
        colsample_bytree=0.8,
    )
    validation_model = LGBMClassifier(**model_params)
    return model_params, validation_model


@app.cell
def _(X_fit, X_valid, log_loss, mo, validation_model, y_fit, y_valid):
    validation_model.fit(X_fit, y_fit)
    validation_prob = validation_model.predict_proba(X_valid)[:, 1]
    validation_loss = log_loss(y_valid, validation_prob)
    mo.md(f"검증 log loss: **{validation_loss:.5f}**")
    return


@app.cell
def _(ConfusionMatrixDisplay, X_valid, plt, validation_model, y_valid):
    fig_confusion, ax_confusion = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_valid, validation_model.predict(X_valid), ax=ax_confusion,
    )
    fig_confusion
    return


@app.cell
def _(features, pl, plt, sns, validation_model):
    importance = pl.DataFrame({
        "Feature": features,
        "Importance": validation_model.feature_importances_,
    }).sort("Importance", descending=True)
    top_features = importance.head(20)
    fig_importance, ax_importance = plt.subplots(figsize=(8, 7))
    sns.barplot(
        x=top_features["Importance"].to_numpy(),
        y=top_features["Feature"].to_list(), ax=ax_importance,
    )
    ax_importance.set(title="Top 20 LightGBM feature importances")
    fig_importance.tight_layout()
    fig_importance
    return


@app.cell
def _(
    LGBMClassifier,
    MinMaxScaler,
    X,
    features,
    model_params,
    np,
    pl,
    test_data,
    y,
):
    final_scaler = MinMaxScaler()
    final_X = final_scaler.fit_transform(X)
    final_test_X = final_scaler.transform(test_data.select(features).to_numpy())
    final_model = LGBMClassifier(**model_params)
    final_model.fit(final_X, y)
    probability = final_model.predict_proba(final_test_X)[:, 1]
    submission = pl.DataFrame({
        "MoleculeId": np.arange(1, len(probability) + 1),
        "PredictedProbability": probability,
    })
    submission.head(10)
    return (submission,)


@app.cell
def _(mo, project_dir, submission):
    submission_path = project_dir / "submission.csv"
    submission.write_csv(submission_path)
    mo.md(f"제출 파일: `{submission_path}` ({submission.height:,}행)")
    return


@app.cell
def _(mo):
    mo.md("""
    Kaggle 제출과 결과 기록은 프로젝트 루트에서 실행합니다.

    ```powershell
    .venv/Scripts/python.exe scripts/kaggle_submit.py notebooks/01_lgbm.py submission.csv
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
