"""학습 스크립트 템플릿."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[3]))
from common.cv import get_folds
from common.utils import seed_everything

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
TARGET = "target"
N_SPLITS = 5
SEED = 42


def main():
    seed_everything(SEED)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    train = get_folds(train, n_splits=N_SPLITS, strategy="stratified", target=TARGET, seed=SEED)

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    for fold in range(N_SPLITS):
        tr = train[train["fold"] != fold]
        va = train[train["fold"] == fold]
        # TODO: 모델 학습
        # model.fit(...)
        # oof[va.index] = model.predict(...)
        # preds += model.predict(test) / N_SPLITS
        print(f"fold {fold}: tr={len(tr)} va={len(va)}")

    # TODO: CV 점수 출력
    SUB_DIR.mkdir(exist_ok=True)
    # pd.DataFrame({"id": test["id"], TARGET: preds}).to_csv(SUB_DIR / "sub_001.csv", index=False)


if __name__ == "__main__":
    main()
