import pandas as pd
from sklearn.base import BaseEstimator
import h2o


class H2ODecorator(BaseEstimator):
    def __init__(self, est_type, est_params, nthreads=-1, mem_max='2G', target_type=None):
        # using H2O estimator classes directly does not work correctly hence string to class mapping is used
        est_map = {
            'dl': h2o.estimators.H2ODeepLearningEstimator,
            'gbm': h2o.estimators.H2OGradientBoostingEstimator,
            'glm': h2o.estimators.H2OGeneralizedLinearEstimator,
            'nb': h2o.estimators.H2ONaiveBayesEstimator,
            'rf': h2o.estimators.H2ORandomForestEstimator,
        }

        self.est_type = est_type
        self.est_params = est_params
        self.est = est_map[est_type](**est_params)
        self.nthreads = nthreads
        self.mem_max = mem_max
        self.cluster_ready = False
        self.target_type = target_type

    def _init_h2o(self):
        if self.cluster_ready:
            return

        h2o.init(nthreads=self.nthreads, max_mem_size=self.mem_max)
        self.cluster_ready = True

    def fit(self, X, y):
        self._init_h2o()

        features = h2o.H2OFrame(python_obj=X)

        if type(y) == pd.Series:
            _y = y.values
        else:
            _y = y

        if self.target_type is not None:
            target_type = [self.target_type]
        else:
            target_type = None

        target = h2o.H2OFrame(python_obj=_y, column_types=target_type)

        self.est.fit(features, target)
        return self

    def predict(self, X):
        self._init_h2o()

        features = h2o.H2OFrame(python_obj=X)
        pred_df = self.est.predict(features).as_data_frame()
        if pred_df.columns.contains('predict'):
            return pred_df['predict']
        else:
            return pred_df.iloc[:, 0]

    def predict_proba(self, X):
        self._init_h2o()

        features = h2o.H2OFrame(python_obj=X)
        pred_df = self.est.predict(features).as_data_frame()
        if pred_df.columns.contains('predict'):
            return pred_df.drop('predict', axis=1).values
        else:
            return pred_df.drop(pred_df.columns[0], axis=1).values
