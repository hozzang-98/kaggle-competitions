import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Biological Response | Fold Ensemble")


@app.cell
def _():
    import time

    import marimo as mo
    import numpy as np
    import polars as pl
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    return (
        CatBoostClassifier,
        LGBMClassifier,
        RandomForestClassifier,
        StratifiedKFold,
        log_loss,
        mo,
        np,
        pl,
        time,
    )


@app.cell
def _(mo):
    mo.md("""
    # Fold 앙상블 비교

    nb01~nb03은 최종 예측을 **전체 데이터로 재학습한 단일 모델**로 만들었다
    (`final_model.fit(X, y)` / `finalize_model`).
    반면 우리 `baseline.py`는 **5개 fold 모델의 평균**으로 제출했고,
    그 XGB가 Private 0.37939로 가장 좋았다.

    여기서는 nb01~03과 같은 모델을 **똑같은 5-fold 앙상블 구조**에 태워
    차이가 모델 때문인지 제출 방식 때문인지 가린다.

    | | 기존 | 여기 |
    |---|---|---|
    | 예측 생성 | 전체 재학습 단일 모델 | 5개 fold 모델 평균 |
    | fold 수 | 3 (PyCaret) / holdout 1회 (01) | 5 |
    | 학습 데이터 | 80% (PyCaret train_size) | 전체 |

    PyCaret 전처리는 쓰지 않는다. EDA에서 확인했듯 이 데이터는 이미 [0,1] 정규화에
    결측도 없어 전처리가 할 일이 없다.
    """)
    return


@app.cell
def _(mo, pl):
    data_dir = mo.notebook_dir().parent / "data"
    project_dir = data_dir.parent

    # polars 기본 스키마 추론은 앞 100행만 본다 → 전체 스캔
    train_data = pl.read_csv(data_dir / "train.csv", infer_schema_length=None)
    test_data = pl.read_csv(data_dir / "test.csv", infer_schema_length=None)

    features = [c for c in train_data.columns if c != "Activity"]
    train_data.shape, test_data.shape
    return features, project_dir, test_data, train_data


@app.cell
def _(features, pl, test_data, train_data):
    # 컬럼 순서 고정 — 트리 모델은 피처를 무작위 샘플링하므로
    # 순서가 바뀌면 random_state가 같아도 결과가 달라진다
    X = train_data.select(features).cast(pl.Float64).to_numpy()
    y = train_data["Activity"].to_numpy()
    X_test = test_data.select(features).cast(pl.Float64).to_numpy()

    assert X.shape[1] == X_test.shape[1] == 1776
    assert X.shape[0] == 3751 and X_test.shape[0] == 2501
    X.shape, X_test.shape
    return X, X_test, y


@app.cell
def _(CatBoostClassifier, LGBMClassifier, RandomForestClassifier):
    SEED = 314      # 원본 노트북들과 같은 시드
    N_FOLD = 5

    # nb01이 쓰던 LightGBM 고정 파라미터 그대로
    lgbm_params = dict(
        random_state=SEED, n_jobs=2, verbosity=-1,
        subsample_freq=20, subsample=0.9,
        reg_lambda=1.2, reg_alpha=1.1,
        num_leaves=200, n_estimators=700,
        min_split_gain=0.4, max_depth=15,
        colsample_bytree=0.8,
    )

    def build_models():
        """fold마다 새로 만든다. nb02의 catboost, nb03의 rf는 PyCaret 기본값에 해당."""
        return {
            "lgbm": LGBMClassifier(**lgbm_params),
            "catboost": CatBoostClassifier(random_state=SEED, verbose=0, thread_count=2),
            "rf": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=2),
        }

    MODELS = ["lgbm", "catboost", "rf"]
    return MODELS, N_FOLD, SEED, build_models, lgbm_params


@app.cell
def _(
    MODELS,
    N_FOLD,
    SEED,
    StratifiedKFold,
    X,
    X_test,
    build_models,
    np,
    time,
    y,
):
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    oof = {m: np.zeros(len(y)) for m in MODELS}
    test_pred = {m: np.zeros(len(X_test)) for m in MODELS}
    fold_scores = {m: [] for m in MODELS}

    from sklearn.metrics import log_loss as _ll

    _t0 = time.time()
    for _k, (_tr, _va) in enumerate(skf.split(X, y), 1):
        _built = build_models()
        for _name in MODELS:
            _m = _built[_name]
            _m.fit(X[_tr], y[_tr])
            oof[_name][_va] = _m.predict_proba(X[_va])[:, 1]
            # 핵심: 전체 재학습 대신 fold 모델들의 평균으로 test를 예측한다
            test_pred[_name] += _m.predict_proba(X_test)[:, 1] / N_FOLD
            fold_scores[_name].append(_ll(y[_va], oof[_name][_va]))
        print(f"fold {_k} 완료 ({time.time() - _t0:.0f}s)")

    # soft blend — 세 모델 확률의 단순 평균 (nb03의 blend_models(method='soft')에 대응)
    oof["blend"] = np.mean([oof[m] for m in MODELS], axis=0)
    test_pred["blend"] = np.mean([test_pred[m] for m in MODELS], axis=0)

    print(f"총 {time.time() - _t0:.0f}s")
    return fold_scores, oof, test_pred


@app.cell
def _(mo):
    mo.md("## OOF LogLoss — fold별 단일 모델 기준")
    return


@app.cell
def _(MODELS, fold_scores, log_loss, np, oof, pl, y):
    rows = []
    for _name in MODELS + ["blend"]:
        rows.append({
            "model": _name,
            "OOF_LogLoss": round(float(log_loss(y, oof[_name])), 5),
            "fold_std": (round(float(np.std(fold_scores[_name])), 5)
                         if _name in fold_scores else None),
        })
    oof_table = pl.DataFrame(rows).sort("OOF_LogLoss")
    oof_table
    return (oof_table,)


@app.cell
def _(mo):
    mo.md("""
    ## 제출 파일 생성

    test.csv에 ID 컬럼이 없다. **행 순서가 곧 MoleculeId** — 섞으면 안 된다.
    """)
    return


@app.cell
def _(X_test, np, pl, project_dir, test_pred):
    out_dir = project_dir / "submissions"
    out_dir.mkdir(exist_ok=True)

    sample = pl.read_csv(project_dir / "data" / "svm_benchmark.csv")
    written = []

    for _name in ["lgbm", "catboost", "blend"]:
        _sub = pl.DataFrame({
            "MoleculeId": np.arange(1, len(X_test) + 1),
            "PredictedProbability": test_pred[_name],
        })
        _path = out_dir / f"fold_ens_{_name}.csv"
        _sub.write_csv(_path)

        # 포맷 검증 — svm_benchmark.csv 가 sample submission 역할
        _chk = pl.read_csv(_path)
        _p = _chk["PredictedProbability"].to_numpy()
        assert _chk.columns == sample.columns, f"컬럼 불일치: {_chk.columns}"
        assert _chk.height == sample.height == 2501, f"행 수 불일치: {_chk.height}"
        assert _chk["MoleculeId"].to_list() == sample["MoleculeId"].to_list()
        assert _chk.null_count().to_numpy().sum() == 0, "결측 있음"
        assert _p.min() >= 0 and _p.max() <= 1, f"확률 범위: [{_p.min()}, {_p.max()}]"
        written.append({
            "file": _path.name, "rows": _chk.height,
            "min": round(float(_p.min()), 4),
            "mean": round(float(_p.mean()), 4),
            "max": round(float(_p.max()), 4),
        })

    pl.DataFrame(written)
    return


@app.cell
def _(mo, oof_table):
    # polars → markdown 직접 작성 (to_pandas는 pyarrow를 요구한다)
    _hdr = "| model | OOF LogLoss | fold std |\n|---|---|---|"
    _body = "\n".join(
        f"| {r['model']} | {r['OOF_LogLoss']:.5f} | "
        f"{'' if r['fold_std'] is None else format(r['fold_std'], '.5f')} |"
        for r in oof_table.to_dicts()
    )

    mo.md(f"""
    ## 정리

    OOF 기준 순위:

    {_hdr}
    {_body}

    OOF는 **fold별 단일 모델**의 점수다. 실제 제출은 **5개 fold 모델의 평균**이라
    이 표가 제출 성능을 그대로 예측하지는 못한다.
    exp01(RF)과 exp02(XGB)에서 CV 순위와 LB 순위가 뒤집힌 것이 그 사례다.

    제출해서 LB로 확인해야 한다.
    """)
    return


if __name__ == "__main__":
    app.run()
