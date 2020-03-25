from functools import partial
from typing import List

import pandas as pd   # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import check_cv  # type: ignore
from sklearn.utils.metaestimators import if_delegate_has_method  # type: ignore
from sklearn.utils import check_array, check_random_state  # type: ignore
from sklearn.base import (  # type: ignore
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier
)
from sklearn.metrics.scorer import check_scoring  # type: ignore
from typing import Tuple, List, Callable, Any


def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False, random_state=None):
    rng = check_random_state(random_state)

    if columns_to_shuffle is None:
        columns_to_shuffle = range(X.shape[1])

    if pre_shuffle:
        X_shuffled = X.copy()
        rng.shuffle(X_shuffled)

    X_res = X.copy()
    for columns in columns_to_shuffle:
        if pre_shuffle:
            X_res[:, columns] = X_shuffled[:, columns]
        else:
            rng.shuffle(X_res[:, columns])
        yield X_res
        X_res[:, columns] = X[:, columns]

def get_score_importances(
        score_func,  # type: Callable[[Any, Any], float]
        X,
        y,
        n_iter=5,  # type: int
        columns_to_shuffle=None,
        random_state=None
    ):
    
    rng = check_random_state(random_state)
    base_score = score_func(X, y)
    scores_decreases = []
    for i in range(n_iter):
        scores_shuffled = _get_scores_shufled(
            score_func, X, y, columns_to_shuffle=columns_to_shuffle,
            random_state=rng
        )
        scores_decreases.append(-scores_shuffled + base_score)
    return base_score, scores_decreases

def _get_scores_shufled(score_func, X, y, columns_to_shuffle=None,
                        random_state=None):
    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state)
    return np.array([score_func(X_shuffled, y) for X_shuffled in Xs])

CAVEATS_CV_NONE = """
Feature importances are computed on the same data as used for training, 
i.e. feature importances don't reflect importance of features for 
generalization.
"""

CAVEATS_CV = """
Feature importances are not computed for the final estimator; 
they are computed for a sequence of estimators trained and evaluated 
on train/test splits. So they tell you about importances of features 
for generalization, but not feature importances of a particular trained model.
"""

CAVEATS_PREFIT = """
If feature importances are computed on the same data as used for training, 
they don't reflect importance of features for generalization. Use a held-out
dataset if you want generalization feature importances.
"""

class PermutationImportance(BaseEstimator, MetaEstimatorMixin):
    
    def __init__(self, estimator, scoring=None, n_iter=5, random_state=None,
                 cv='prefit', refit=True):
        # type: (...) -> None
        if isinstance(cv, str) and cv != "prefit":
            raise ValueError("Invalid cv value: {!r}".format(cv))
        self.refit = refit
        self.estimator = estimator
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        self.rng_ = check_random_state(random_state)

    def _wrap_scorer(self, base_scorer, pd_columns):
        def pd_scorer(model, X, y):
            X = pd.DataFrame(X, columns=pd_columns)
            return base_scorer(model, X, y)
        return pd_scorer

    def fit(self, X, y, groups=None, **fit_params):
        
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if isinstance(X, pd.DataFrame):
            self.scorer_ = self._wrap_scorer(self.scorer_, X.columns)

        if self.cv != "prefit" and self.refit:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **fit_params)

        X = check_array(X)

        if self.cv not in (None, "prefit"):
            si = self._cv_scores_importances(X, y, groups=groups, **fit_params)
        else:
            si = self._non_cv_scores_importances(X, y)
        scores, results = si
        self.scores_ = np.array(scores)
        self.results_ = results
        self.feature_importances_ = np.mean(results, axis=0)
        self.feature_importances_std_ = np.std(results, axis=0)
        return self

    def _cv_scores_importances(self, X, y, groups=None, **fit_params):
        assert self.cv is not None
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        feature_importances = []  # type: List
        base_scores = []  # type: List[float]
        for train, test in cv.split(X, y, groups):
            est = clone(self.estimator).fit(X[train], y[train], **fit_params)
            score_func = partial(self.scorer_, est)
            _base_score, _importances = self._get_score_importances(
                score_func, X[test], y[test])
            base_scores.extend([_base_score] * len(_importances))
            feature_importances.extend(_importances)
        return base_scores, feature_importances

    def _non_cv_scores_importances(self, X, y):
        score_func = partial(self.scorer_, self.wrapped_estimator_)
        base_score, importances = self._get_score_importances(score_func, X, y)
        return [base_score] * len(importances), importances

    def _get_score_importances(self, score_func, X, y):
        return get_score_importances(score_func, X, y, n_iter=self.n_iter,
                                     random_state=self.rng_)
    
    @property
    def caveats_(self):
        # type: () -> str
        if self.cv == 'prefit':
            return CAVEATS_PREFIT
        elif self.cv is None:
            return CAVEATS_CV_NONE
        return CAVEATS_CV

    # ============= Exposed methods of a wrapped estimator:

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def score(self, X, y=None, *args, **kwargs):
        return self.wrapped_estimator_.score(X, y, *args, **kwargs)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict(self, X):
        return self.wrapped_estimator_.predict(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_proba(self, X):
        return self.wrapped_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_log_proba(self, X):
        return self.wrapped_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def decision_function(self, X):
        return self.wrapped_estimator_.decision_function(X)

    @property
    def wrapped_estimator_(self):
        if self.cv == "prefit" or not self.refit:
            return self.estimator
        return self.estimator_

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.wrapped_estimator_.classes_