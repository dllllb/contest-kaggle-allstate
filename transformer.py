from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


class TargetCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, builder, columns=None, n_jobs=1, true_label=None):
        self.vc = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.true_label = true_label
        self.builder = builder

    def fit(self, df, y=None):
        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        if self.true_label is not None:
            target = (y == self.true_label)
        else:
            target = y

        encoders = Parallel(n_jobs=self.n_jobs)(
            delayed(self.builder)(df[col], target)
            for col in columns
        )

        self.vc = dict(zip(columns, encoders))

        return self

    def transform(self, df):
        res = df.copy()
        for col, encoder in self.vc.items():
            res[col] = encoder(res[col])
        return res


def map_encoder(vals, mapping):
    return vals.map(lambda x: mapping.get(x, mapping.get('nan', 0)))


def build_zeroing_encoder(column, __, threshold, top, placeholder):
    vc = column.replace(np.nan, 'nan').value_counts()
    candidates = set(vc[vc <= threshold].index).union(set(vc[top:].index))
    mapping = dict(zip(vc.index, vc.index))
    if 'nan' in mapping:
        mapping['nan'] = np.nan
    for c in candidates:
        mapping[c] = placeholder
    return partial(map_encoder, mapping=mapping)


def high_cardinality_zeroing(threshold=1, top=10000, placeholder='zeroed', columns=None, n_jobs=1):
    buider = partial(
        build_zeroing_encoder,
        threshold=threshold,
        top=top,
        placeholder=placeholder
    )

    return TargetCategoryEncoder(buider, columns, n_jobs)


def build_categorical_feature_encoder_mean(column, target, size_threshold):
    global_mean = target.mean()
    col_dna = column.fillna('nan')
    means = target.groupby(col_dna).mean()
    counts = col_dna.groupby(col_dna).count()
    reg = pd.DataFrame(counts / (counts + size_threshold))
    reg[1] = 1.
    reg = reg.min(axis=1)
    means_reg = means * reg + (1-reg) * global_mean
    entries = means_reg.sort_values(ascending=False).index

    mapping = dict(zip(entries, range(len(entries))))
    encoder = partial(map_encoder, mapping=mapping)
    return encoder


def target_mean_encoder(columns=None, n_jobs=1, size_threshold=10, true_label=None):
    buider = partial(
        build_categorical_feature_encoder_mean,
        size_threshold=size_threshold
    )
    return TargetCategoryEncoder(buider, columns, n_jobs, true_label)
