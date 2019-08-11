import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor

import transformer as tr


def update_model_stats(stats_file, params, results):
    import json
    import os.path
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []
        
    stats.append({**results, **params})
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

        
def run_experiment(evaluator, params, stats_file):    
    import time
    
    params = init_params(params)
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})


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


def cv_test(est, n_folds):
    df = pd.read_csv('train.csv.gz', index_col='id')

    features = df.drop('loss', axis=1)
    target = df.loss

    if type(est) is tuple:
        transform, estimator = est
        features_t = transform.fit_transform(features, target)
    else:
        estimator = est
        features_t = features

    cv = KFold(n_folds, shuffle=True)

    scores = cross_val_score(estimator, X=features_t, y=target, scoring=make_scorer(mape), cv=cv)
    return {'mape-mean': scores.mean(), 'mape-std': scores.std()}
    
    
def init_params(overrides):
    defaults = {
        'validation-type': 'cv',
        'n_folds': 3,
        'target_distr': 'normal'
    }
    return {**defaults, **overrides}


def pred_vs_true(est, path):
    df = pd.read_csv('train.csv.gz', index_col='id')
    features = df.drop('loss', axis=1)
    target = df.loss.values

    transform, estimator = est
    pl = make_pipeline(transform, estimator)

    splits = train_test_split(features, target, train_size=0.9, random_state=123)
    x_train, x_test, y_train, y_test = splits
    y_pred = pl.fit(x_train, y_train).predict(x_test)
    pd.DataFrame({'pred': y_pred, 'true': y_test}).to_csv(path, index=False, sep='\t')
    
    
def init_xbg_est(params):
    keys = {
        'learning_rate',
        'n_estimators',
        'max_depth',
        'min_child_weight',
        'subsample',
        'colsample_bytree',
    }
    
    obj = "reg:linear" if params['target_distr'] != 'poisson' else 'count:poisson'

    xgb_params = {
        "objective": obj,
        **{k: v for k, v in params.items() if k in keys},
    }

    class XGBC(XGBRegressor):
        def fit(self, x, y, **kwargs):
            f_train, f_val, t_train, t_val = train_test_split(x, y, test_size=params['es_share'])
            super().fit(
                f_train,
                t_train,
                eval_set=[(f_val, t_val)],
                eval_metric=mape_evalerror_exp,
                early_stopping_rounds=params['num_es_rounds'],
                verbose=120)

    return XGBC(**xgb_params)


def init_h2o_est(params): 
    h2o_gbm_params = {
        'model_id': 'kaggle_allstate_gbm',
        'distribution': params['target_distr'],
        'ntrees': params['num_rounds'],
        'learn_rate': params['eta'],
        'max_depth': params['max_depth'],
        'sample_rate': params['subsample'],
        'col_sample_rate_per_tree': params['colsample_bytree']
    }
    
    import h2o_est as h2
    return h2.H2ODecorator('gbm', h2o_gbm_params)


def validate(params):    
    category_encoding = params['category_encoding']
    
    if category_encoding == 'onehot':
        transf = make_pipeline(
            tr.high_cardinality_zeroing(threshold=50),
            tr.df2dict(),
            DictVectorizer(sparse=False)
        )
    elif category_encoding == 'target_mean':
        transf = tr.target_mean_encoder(size_threshold=20)
    elif category_encoding == 'none':
        pass
    else:
        raise AssertionError(f'unknown category encoding type: {category_encoding}')

    est_type = params['est_type']
    if est_type == 'h2o':
        est = init_h2o_est(params)
    elif est_type == 'xgb':
        est = init_xbg_est(params)
    else:
        raise AssertionError(f'unknown estimator type: {est_type}')
    
    if params['target_log']:
        est = TargetTransfRegressor(est, np.log, np.exp)
        
    if category_encoding == 'none':
        pl = est
    else:
        pl = make_pipeline(transf, est)
    return cv_test(pl, n_folds=params['n_folds'])


def test_validate():
    params = {
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 6,
        'subsample': .1,
        'colsample_bytree': .2,
        'category_encoding': 'target_mean',
        'num_rounds': 10,
        'est_type': 'xgb',
        'num_es_rounds': None,
        'es_share': .1,
        'target_log': True,
    }
    print(validate(init_params(params)))
