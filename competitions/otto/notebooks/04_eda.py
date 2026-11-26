import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | EDA")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import math

    return math, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Otto EDA

    목적은 관찰 나열이 아니라 **베이스라인 설계 결정**이다.
    각 절 끝에 "그래서 무엇을 바꾸는가"를 적는다.
    데이터는 `01_make_parquet.py`가 만든 `data/*.parquet`를 쓴다.
    """)
    return


@app.cell
def _(mo, pl):
    data_dir = mo.notebook_dir().parent / "data"
    train = pl.read_parquet(data_dir / "train.parquet")
    test = pl.read_parquet(data_dir / "test.parquet")
    feats = [c for c in train.columns if c.startswith("feat_")]
    {"train": train.shape, "test": test.shape, "n_feats": len(feats)}
    return feats, test, train


@app.cell
def _(feats, np, test, train):
    X = train.select(feats).to_numpy().astype(np.float64)
    X_test = test.select(feats).to_numpy().astype(np.float64)
    y = train["target"].to_physical().to_numpy()
    class_names = train["target"].dtype.categories.to_list()
    return X, X_test, class_names, y


@app.cell
def _(mo):
    mo.md("""## 1. 클래스 분포 — 불균형 8.4배""")
    return


@app.cell
def _(class_names, np, pl, train, y):
    _n = len(y)
    _cnt = np.bincount(y, minlength=9)
    dist = pl.DataFrame({
        "class": class_names,
        "count": _cnt,
        "ratio": (_cnt / _n).round(4),
        "bar": ["#" * int(c / _n * 100) for c in _cnt],
    })
    dist
    return (dist,)


@app.cell
def _(mo, np, y):
    _pc = np.bincount(y, minlength=9) / len(y)
    mo.md(f"""
    최대 **Class_2 {_pc.max()*100:.2f}%** / 최소 **Class_1 {_pc.min()*100:.2f}%** → **{_pc.max()/_pc.min():.2f}배**.
    Class_2+Class_6이 전체의 **{(_pc[1]+_pc[5])*100:.1f}%**.

    > **결정**: 층화(stratified) 분할이 필수. 단순 random split이면 fold마다
    > Class_1(1,929행)의 비율이 흔들려 CV가 불안정해진다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 2. 기준점 — 2.19722 / 1.95041""")
    return


@app.cell
def _(math, mo, np, y):
    uniform = math.log(9)
    _pc = np.bincount(y, minlength=9) / len(y)
    entropy = float(-(_pc * np.log(_pc)).sum())
    mo.md(f"""
    | 기준점 | 값 | 의미 |
    |---|---|---|
    | 균등 1/9 (`ln 9`) | **{uniform:.5f}** | 이보다 나쁘면 파이프라인 버그 |
    | 클래스 분포 엔트로피 | **{entropy:.5f}** | 피처를 하나도 안 쓴 점수 |
    | 차이 | {uniform - entropy:.5f} | 불균형이 만든 공짜 이득 |

    > **결정**: 모델 점수를 **{entropy:.4f}와 먼저 비교**한다. 이걸 못 이기면
    > 93개 피처가 일을 안 한 것이다. 균등({uniform:.4f})은 통과선이 아니라 버그 탐지선이다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 3. 결측 · 상수 · 중복 — 전부 0""")
    return


@app.cell
def _(X, feats, mo, np, test, train):
    _nulls = train.null_count().sum_horizontal().item() + test.null_count().sum_horizontal().item()
    _nuniq = np.array([len(np.unique(X[:, i])) for i in range(X.shape[1])])
    _const = int((_nuniq == 1).sum())
    _seen, _dup = {}, 0
    for _i in range(X.shape[1]):
        _k = X[:, _i].tobytes()
        if _k in _seen:
            _dup += 1
        else:
            _seen[_k] = _i
    mo.md(f"""
    | 항목 | 수 |
    |---|---|
    | 결측 (train+test) | **{_nulls}** |
    | 상수 피처 | **{_const}** |
    | 완전중복 피처 | **{_dup}** |
    | 피처 수 | {len(feats)} |

    > **결정**: 결측 처리·상수 제거·중복 제거 코드를 **전부 생략한다.**
    > Bioresponse와 같은 결론이며, 주최측이 정리해서 배포한 데이터다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 4. 희소도 — 셀의 79%가 0""")
    return


@app.cell
def _(X, X_test, mo, np):
    _nz = (X > 0).sum(1)
    _rs = X.sum(1)
    mo.md(f"""
    | 항목 | 값 |
    |---|---|
    | train 0 비율 | **{(X == 0).mean()*100:.2f}%** |
    | test 0 비율 | {(X_test == 0).mean()*100:.2f}% |
    | 행별 non-zero 개수 | min {_nz.min()} / 중앙 {np.median(_nz):.0f} / max {_nz.max()} (전체 93) |
    | 행 합계(총 카운트) | min {_rs.min():.0f} / 중앙 {np.median(_rs):.0f} / max {_rs.max():.0f} |
    | 전부 0인 행 | {(_nz == 0).sum()} |

    값은 **0 이상 정수(최대 398)**, 음수 없음 → 카운트 데이터 확정.
    93칸 중 중앙값 18칸만 채워져 있다.

    > **결정**: 행별 `non-zero 개수`와 `행 합계`는 **그 자체가 피처 후보**다.
    > 클래스마다 다른지는 6절에서 확인한다.
    > 다만 베이스라인은 원본 93개로 먼저 돌린다 — 변수를 하나만 바꾸기 위해서.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 5. 피처 간 상관 — |r|>0.9 가 0쌍""")
    return


@app.cell
def _(X, feats, mo, np):
    _Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    _C = (_Z.T @ _Z) / len(_Z)
    _iu = np.triu_indices(X.shape[1], 1)
    _r = _C[_iu]
    _top = np.argsort(-np.abs(_r))[:5]
    _rows = "\n".join(
        f"| {feats[_iu[0][k]]} | {feats[_iu[1][k]]} | {_r[k]:.3f} |" for k in _top
    )
    mo.md(f"""
    전체 {len(_r):,}쌍 중 최대 |r| = **{np.abs(_r).max():.4f}**

    | 임계 | 쌍 수 |
    |---|---|
    | \\|r\\|>0.9 | **{(np.abs(_r)>0.9).sum()}** |
    | \\|r\\|>0.8 | {(np.abs(_r)>0.8).sum()} |
    | \\|r\\|>0.7 | {(np.abs(_r)>0.7).sum()} |
    | \\|r\\|>0.5 | {(np.abs(_r)>0.5).sum()} |

    | 피처 A | 피처 B | r |
    |---|---|---|
    {_rows}

    > **결정**: 상관 기반 피처 제거 **불필요**. Bioresponse와 동일하게
    > 주최측이 이미 정리한 것으로 보인다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 6. 타깃 신호 — 피처별 eta²""")
    return


@app.cell
def _(X, feats, mo, np, y):
    _Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    _w = np.bincount(y, minlength=9) / len(y)
    eta2 = np.array([
        sum(_w[c] * (_Z[y == c, i].mean() - _Z[:, i].mean())**2 for c in range(9))
        / (_Z[:, i].var() + 1e-12)
        for i in range(X.shape[1])
    ])
    _o = np.argsort(-eta2)
    _rows = "\n".join(f"| {feats[i]} | {eta2[i]:.4f} |" for i in _o[:8])
    mo.md(f"""
    클래스별 평균이 전체 분산을 얼마나 설명하는가(0~1). 높을수록 그 피처 혼자로 클래스가 갈린다.

    | 피처 | eta² |
    |---|---|
    {_rows}

    - eta² > 0.05 인 피처: **{(eta2>0.05).sum()} / {len(feats)}**
    - 최저: {feats[_o[-1]]} ({eta2[_o[-1]]:.4f})

    > **관찰**: `feat_34`가 **{eta2[_o[0]]:.2f}** 로 압도적이다. 단일 피처가 이 정도면
    > 특정 클래스 전용 마커일 가능성이 높다 → 7절에서 확인.
    > Bioresponse(약한 신호가 넓게 분산)와 달리 **강한 신호가 소수에 집중**돼 있다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 7. 클래스별 성격 — 어디서 틀릴 것인가""")
    return


@app.cell
def _(X, class_names, np, pl, y):
    _rows = []
    for _c in range(9):
        _S = X[y == _c]
        _rows.append({
            "class": class_names[_c],
            "n": len(_S),
            "nonzero_median": float(np.median((_S > 0).sum(1))),
            "rowsum_median": float(np.median(_S.sum(1))),
            "feat_34_mean": round(float(_S[:, 33].mean()), 2),
            "feat_34_nonzero%": round(float((_S[:, 33] > 0).mean() * 100), 1),
        })
    per_class = pl.DataFrame(_rows)
    per_class
    return (per_class,)


@app.cell
def _(X, class_names, np, pl, y):
    _M = np.array([X[y == c].mean(0) for c in range(9)])
    _Mz = (_M - _M.mean(1, keepdims=True)) / (_M.std(1, keepdims=True) + 1e-12)
    _P = (_Mz @ _Mz.T) / X.shape[1]
    profile_corr = pl.DataFrame(
        {"class": class_names,
         **{class_names[j]: _P[:, j].round(2) for j in range(9)}}
    )
    profile_corr
    return (profile_corr,)


@app.cell
def _(mo):
    mo.md("""
    **feat_34는 Class_5 전용 마커다.** Class_5만 non-zero 99%·평균 13.5,
    나머지 8개 클래스는 전부 평균 0.3 근처. Class_5는 쉽게 분리된다.

    **Class_2 / Class_3 / Class_4가 서로 닮았다** (프로파일 상관 0.79~0.93).
    셋 합쳐 train의 43%이고, 여기서 LogLoss 손실이 집중될 것이다.

    희소도도 클래스마다 다르다 — Class_5는 non-zero 중앙값 8개, Class_6은 24개.

    > **결정 1**: 행별 non-zero 개수·행 합계를 피처로 추가하는 실험을 **별도 변수 1개로** 돌릴 가치가 있다.
    > **결정 2**: 모델 평가 시 전체 LogLoss만 보지 말고 **클래스별 LogLoss 기여**를 쪼개 본다.
    > Class_2/3/4에서 얼마나 잃는지가 개선 지점이다.
    > **결정 3**: 소수 클래스(Class_1, 1,929행)에서 확률 0이 나오기 쉽다 → **clip 필수**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 8. train vs test 분포 차이 — 없음""")
    return


@app.cell
def _(X, X_test, feats, mo):
    _d = sorted(
        ((c, X[:, i].mean(), X_test[:, i].mean()) for i, c in enumerate(feats)),
        key=lambda t: -abs(t[1] - t[2]) / (t[1] + 1e-9),
    )
    _rows = "\n".join(
        f"| {c} | {a:.3f} | {b:.3f} | {abs(a-b)/(a+1e-9)*100:.1f}% |" for c, a, b in _d[:5]
    )
    _over = sum(1 for c, a, b in _d if abs(a - b) / (a + 1e-9) > 0.1)
    mo.md(f"""
    | 피처 | train 평균 | test 평균 | 상대차 |
    |---|---|---|---|
    {_rows}

    상대차 10% 초과 피처: **{_over}개**

    > **결정**: 분포 이동(drift)이 사실상 없다. adversarial validation 불필요.
    > **CV가 LB를 어느 정도 반영할 것으로 기대할 수 있다** — 다만 Bioresponse에서
    > "갭은 모델마다 다르다"를 배웠으므로, 한 모델로 재고 일반화하지 않는다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 종합 — 베이스라인 설계에 반영할 것

    | 관찰 | 설계 결정 |
    |---|---|
    | 결측·상수·중복 0, \\|r\\|>0.9 0쌍 | 전처리 코드 **전부 생략** |
    | 클래스 불균형 8.4배 | **StratifiedKFold** 필수 |
    | 분포 엔트로피 1.9504 | 모델 점수를 **이것과 먼저 비교** |
    | 소수 클래스 1,929행 | 확률 **clip 필수** (0 → ∞) |
    | 카운트 + 79% 희소 | 트리 모델이 1순위. `log1p`·행집계는 **별도 실험 1변수**로 |
    | feat_34 = Class_5 마커 | Class_5는 쉬움. 여기서 점수가 갈리지 않는다 |
    | Class_2/3/4 닮음 (43%) | **클래스별 LogLoss 기여**를 쪼개서 본다 |
    | train/test drift 없음 | adversarial validation 불필요 |

    다음: `02_baseline.py` 원본 실행(holdout) → 같은 4모델 StratifiedKFold OOF 재측정 → 순위 비교.
    """)
    return


if __name__ == "__main__":
    app.run()
