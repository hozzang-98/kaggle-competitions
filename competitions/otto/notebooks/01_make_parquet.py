import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Otto | Make Parquet")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Otto: CSV → parquet

    `data/raw/*.csv`를 읽어 열마다 dtype·min·max·고유값 수·null을 진단하고,
    대회에 맞는 형으로 바꿔 `data/*.parquet`로 저장합니다.
    형 결정은 train·test를 합친 범위로 합니다(test 최대값이 train보다 클 수 있음).
    """)
    return


@app.cell
def _(mo, pl):
    data_dir = mo.notebook_dir().parent / "data"
    raw_dir = data_dir / "raw"
    raw = {p.stem: pl.read_csv(p) for p in sorted(raw_dir.glob("*.csv"))}
    {name: df.shape for name, df in raw.items()}
    return data_dir, raw, raw_dir


@app.cell
def _(pl):
    UINT = [(pl.UInt8, 2**8 - 1), (pl.UInt16, 2**16 - 1), (pl.UInt32, 2**32 - 1)]
    INT = [(pl.Int8, 2**7 - 1), (pl.Int16, 2**15 - 1), (pl.Int32, 2**31 - 1)]

    def propose(s: pl.Series):
        """가장 작은 무손실 형을 제안한다. float·null 처리는 대회별로 판단."""
        if s.dtype == pl.String:
            return pl.Categorical if s.n_unique() / len(s) < 0.5 else pl.String
        if not s.dtype.is_integer():
            return s.dtype
        lo, hi = s.min(), s.max()
        for dtype, top in UINT if lo >= 0 else INT:
            if max(hi, -lo - 1) <= top:
                return dtype
        return pl.Int64

    def diagnose(df: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for c in df.columns:
            s = df[c]
            numeric = s.dtype.is_numeric()
            rows.append({
                "column": c,
                "dtype": str(s.dtype),
                "min": str(s.min()) if numeric else "",
                "max": str(s.max()) if numeric else "",
                "n_unique": s.n_unique(),
                "nulls": s.null_count(),
                "proposed": str(propose(s.drop_nulls())),
            })
        return pl.DataFrame(rows)

    return diagnose, propose


@app.cell
def _(mo):
    mo.md("""
    ## 열 진단
    """)
    return


@app.cell
def _(diagnose, mo, raw):
    mo.ui.tabs({name: mo.ui.table(diagnose(df), page_size=20) for name, df in raw.items()})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Otto 형 결정

    - `id`: train·test 범위 → 제안 형
    - `feat_*`: 음수·null 없는 개수형 → train·test 합친 max로 제안 형(`UInt8`/`UInt16`)
    - `target`: `Class_1`~`Class_9` → `pl.Enum` (학습 때 `to_physical()`로 0~8)
    - sampleSubmission `Class_*`: 확률 자리 → `Float32`
    """)
    return


@app.cell
def _(pl, propose, raw):
    train_raw, test_raw, sample_raw = raw["train"], raw["test"], raw["sampleSubmission"]
    shared = [c for c in test_raw.columns if c in train_raw.columns]
    both = pl.concat([train_raw.select(shared), test_raw.select(shared)])
    schema = {c: propose(both[c]) for c in shared}

    classes = sorted(train_raw["target"].unique().to_list())
    target_enum = pl.Enum(classes)

    train = train_raw.cast({**schema, "target": target_enum})
    test = test_raw.cast(schema)
    sample = sample_raw.cast(
        {"id": schema["id"], **{c: pl.Float32 for c in sample_raw.columns if c != "id"}}
    )
    {str(k): sum(1 for v in schema.values() if v == k) for k in set(schema.values())}
    return sample, test, train


@app.cell
def _(mo):
    mo.md("""
    ## 저장과 대조
    """)
    return


@app.cell
def _(data_dir, pl, raw, raw_dir, sample, test, train):
    out = {"train": train, "test": test, "sampleSubmission": sample}
    report = []
    for _name, _df in out.items():
        _path = data_dir / f"{_name}.parquet"
        _df.write_parquet(_path)
        _back = pl.read_parquet(_path)
        assert _back.shape == raw[_name].shape, _name
        assert _back.equals(_df), _name
        report.append({
            "file": _path.name,
            "rows": _back.height,
            "cols": _back.width,
            "csv_MB": round((raw_dir / f"{_name}.csv").stat().st_size / 2**20, 2),
            "parquet_MB": round(_path.stat().st_size / 2**20, 2),
            "mem_MB_before": round(raw[_name].estimated_size("mb"), 2),
            "mem_MB_after": round(_back.estimated_size("mb"), 2),
        })
    pl.DataFrame(report)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
