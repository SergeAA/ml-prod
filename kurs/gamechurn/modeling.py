import logging
import pickle
from os.path import exists
from time import time
import pandas as pd

import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import f1_score

from .tools import loadDF


class Model:
    """Create model"""

    def __init__(self, data: pd.DataFrame, model_folder: str, params=None):
        """Init model with specific parameters
        """

        self.model_file = model_folder+'/model.pkl'
        self.features_file = model_folder+'/features.pkl'
        self.model = None
        self.features = None

        if params:
            self.params = params
        else:
            self.params = {
                'max_depth': 3,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'nthread': 5,
                'subsample': 1.,
                'colsample_bytree': 0.5,
                'min_child_weight ':  3,
                'reg_alpha': 0.,
                'reg_lambda': 0.,
                'seed': 42,
                'missing': 1e10
            }

        self.__loadModel()

        X = data.drop(['user_id', 'is_churned'], axis=1)
        self.columns = X.columns.to_list()
        self.y = data['is_churned']
        self.X = MinMaxScaler().fit_transform(X)

    def __getBalanced(self, X, y, ratio=0.3):
        return SMOTE(random_state=42, ratio=ratio).fit_sample(X, y)

    def getTrainTest(self, test_size=0.3, disbalance_ratio=0.3, features=None):
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=test_size,
                             shuffle=True, stratify=self.y, random_state=42)
        X_train, y_train = self.__getBalanced(X_train, y_train, ratio=disbalance_ratio)

        X_train = self.__getX(X_train, features)
        X_test = self.__getX(X_test, features)
        return (X_train, X_test, y_train, y_test)

    def __getX(self, X, features=None):
        X = pd.DataFrame(X, columns=self.columns)
        if not features:
            features = self.features
        return X[features] if features else X

    def selectFeatures(self, kind='corr', **args):
        if kind != 'corr':
            raise NotImplemented

        df = pd.DataFrame(self.X, columns=self.columns)
        df['target'] = self.y
        corr = df.corr()
        thr = args.get('threshold', 0.01)
        target = corr['target']
        target = target[target.abs() > thr].abs().drop('target').sort_values(ascending=False)
        self.features = target.index.to_list()
        return target

    def __testModel(self, model):
        X_train, X_test, y_train, y_test = self.getTrainTest()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_true=y_test, y_pred=y_pred)

    def baseline(self):
        return self.__testModel(LogisticRegression(random_state=42))

    def test(self):
        return self.__testModel(xgb.XGBClassifier(**self.params))

    def crossValidate(self, n_cv=3):
        md = xgb.XGBClassifier(**self.params)
        X, y = self.__getBalanced(self.__getX(self.X), self.y)
        res = cross_validate(md, X, y, scoring='f1',
                             cv=StratifiedKFold(n_cv, random_state=42), n_jobs=-1)
        return sum(res['test_score']) / len(res['test_score'])

    def selectParams(self, grid={'n_estimators': [10, 20]}, n_cv=2):
        md = xgb.XGBClassifier(**self.params)
        X, y = self.__getBalanced(self.__getX(self.X), self.y)
        clf = RandomizedSearchCV(md, grid, random_state=42, n_jobs=-1,
                                 cv=StratifiedKFold(n_cv, random_state=42), scoring='f1')
        clf.fit(X, y)
        self.params.update(clf.best_params_)
        return clf.best_score_

    def __saveModel(self):
        with open(self.model_file, 'wb') as mdf, open(self.features_file, 'wb') as ff:
            pickle.dump(self.model, mdf)
            pickle.dump(self.features, ff)

    def __loadModel(self):
        if exists(self.model_file) and exists(self.features_file):
            logging.info('Model already exists ... loading')
            with open(self.model_file, 'rb') as mdf, open(self.features_file, 'rb') as ff:
                self.model = pickle.load(mdf)
                self.features = pickle.load(ff)

    def fit_save(self):
        self.model = xgb.XGBClassifier(**self.params)
        X, y = self.__getBalanced(self.__getX(self.X), self.y)
        self.model.fit(X, y)
        self.__saveModel()
        return self.model

    def predict(self, test):
        if not self.model:
            raise BaseException('Model is not fit or loaded')
        X_test = MinMaxScaler().fit_transform(test[self.features])
        return self.model.predict(X_test)

    def predict_proba(self, test):
        if not self.model:
            raise BaseException('Model is not fit or loaded')
        X_test = MinMaxScaler().fit_transform(test[self.features])
        return self.model.predict_proba(X_test)[:, 1]
