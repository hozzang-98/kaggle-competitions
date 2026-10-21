"""교차검증 분할 헬퍼."""
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def get_folds(df, n_splits=5, strategy="stratified", target=None, group=None, seed=42):
    """fold 인덱스를 담은 컬럼을 df에 추가해서 반환.

    strategy: "kfold" | "stratified" | "group"
    """
    df = df.copy()
    df["fold"] = -1

    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(df, df[target])
    elif strategy == "group":
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(df, df[target] if target else None, df[group])
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(df)

    for fold, (_, valid_idx) in enumerate(splits):
        df.loc[df.index[valid_idx], "fold"] = fold
    return df
