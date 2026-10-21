"""대회 공통 유틸."""
import os
import random

import numpy as np
import pandas as pd


def seed_everything(seed: int = 42) -> None:
    """재현성을 위한 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """수치형 컬럼 다운캐스팅으로 메모리 절감."""
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"메모리: {start:.1f}MB -> {end:.1f}MB ({100 * (start - end) / start:.1f}% 감소)")
    return df
