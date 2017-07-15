import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator

class TargetTransfRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_est, transf_to, transf_from):
        self.base_est = base_est
        self.transf_to = transf_to
        self.transf_from = transf_from

    def fit(self, X, y):
        self.base_est.fit(X, self.transf_to(y))
        return self

    def predict(self, X):
        return self.transf_from(self.base_est.predict(X))


def mape(y_true, y_pred):
    return np.average(np.abs(y_pred - y_true), axis=0)


def mape_evalerror_exp(preds, dtrain):
    res = np.average(np.abs(np.exp(preds) - np.exp(dtrain.get_label())), axis=0)
    return 'mae', res


def mape_evalerror(preds, dtrain):
    return 'mape', mape(dtrain.get_label(), preds)


def ybin(y):
    return (y.astype(np.float64) / np.max(y) * 10).astype(np.byte)
