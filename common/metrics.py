"""자주 쓰는 평가 지표."""
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rmsle(y_true, y_pred):
    return rmse(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))


def auc(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))
