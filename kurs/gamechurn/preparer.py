import logging
from re import sub
from os.path import exists
from datetime import datetime
from time import time
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .tools import loadDF, saveDF


class Preparer:
    def __init__(self, features={}, fillNA=0, exclude=[], clusters=4, compress=3, compress_threshold=0.05):
        if fillNA is not None:
            self.fillNA = fillNA
        self.features = features
        self.clusters = clusters
        self.compress = compress
        self.compress_threshold = compress_threshold
        self.exclude = ['user_id', 'is_churned']
        if exclude:
            self.exclude.append(exclude)
        self.__cache = {
            'median': {},
            'mean': {},
            'mode': {},
        }

    def median(self, df, col):
        self.__cache['median'][col] = self.__cache['median'].get(col, df[col].median())
        return self.__cache['median'][col]

    def mean(self, df, col):
        self.__cache['mean'][col] = self.__cache['mean'].get(col, df[col].mean())
        return self.__cache['mean'][col]

    def mode(self, df, col):
        self.__cache['mode'][col] = self.__cache['mode'].get(col, df[col].mode()[0])
        return self.__cache['mode'][col]

    def quantile(self, df, col, q=0.25):
        qq = f'q-{q}'
        if not self.__cache.get(qq, None):
            self.__cache[qq] = {}
        self.__cache[qq][col] = self.__cache[qq].get(col, df[col].quantile(q))
        return self.__cache[qq][col]

    def fillMedian(self, df, col):
        df[col] = df[col].fillna(self.median(df, col))

    def fillMean(self, df, col):
        df[col] = df[col].fillna(self.mean(df, col))

    def fillMode(self, df, col):
        df[col] = df[col].fillna(self.mode(df, col))

    def updateByQuantile(self, df, col, value, q=(0.25, 0.75), th=1.5):
        Q1 = self.quantile(df, col, q[0])
        Q3 = self.quantile(df, col, q[1])
        IQR = Q3 - Q1
        flt = (df[col] < (Q1 - th * IQR)) | (df[col] > (Q3 + th * IQR))
        df.loc[flt, col] = value

    def updateInQuantile(self, df, col, qMin=0.05, qMax=0.95):
        if qMin:
            value = self.quantile(df, col, qMin)
            self.updateByRange(df, col, value, minValue=value)
        if qMax:
            value = self.quantile(df, col, qMax)
            self.updateByRange(df, col, value, maxValue=value)

    def updateByRange(self, df, col, value, minValue=None, maxValue=None):
        if minValue is None and maxValue is None:
            return
        if minValue is not None:
            flt = df[col] < minValue
            if maxValue is not None:
                flt |= df[col] > maxValue
        else:
            flt = df[col] > maxValue

        df.loc[flt, col] = value

    def _age(self, df, col):
        md = round(self.median(df, col))
        df[col] = df[col].fillna(md)
        # self.updateByRange(df, col, md, 7, 80)
        self.updateByQuantile(df, col, md, q=(0.05, 0.95), th=0)

    def _gender(self, df, col):
        self.fillMode(df, col)
        df.loc[~df[col].isin(['M', 'F']), col] = self.mode(df, col)
        df[col] = df[col].map({'M': 1., 'F': 0.})

    def _days_between_reg_fl(self, df, col):
        self.updateByRange(df, col, 0, minValue=0)

    def _days_between_fl_df(self, df, col):
        self.updateByRange(df, col, -1, minValue=0)

    def _avg_min_ping(self, df, col):
        self.fillMedian(df, col)
        self.updateByRange(df, col, self.median(df, col), 0)

    def __default_outliner(self, df, col):
        self.fillMedian(df, col)
        self.updateInQuantile(df, col, qMin=None, qMax=0.95)

    def _core_type_nm(self, dataset, col):
        exc = [col]
        dataset[col] = dataset[col].fillna('Other')
        for i in ['Core', 'Returns', 'Mantle', 'Other']:
            nf = f'Type_{i}'
            exc.append(nf)
            dataset[nf] = dataset[col] == i
            dataset[nf] = dataset[nf].astype(int)
        dataset.drop(col, axis=1, inplace=True)
        return exc

    def createFeatures(self, df, flds):
        logging.debug('\t\tcreating %d clusters', self.clusters)
        if not self.__cache.get('KMEANS', None):
            self.__cache['KMEANS'] = KMeans(n_clusters=self.clusters, random_state=42,
                                            precompute_distances=False, n_jobs=-1)
            dd = self.__cache['KMEANS'].fit_predict(df[flds])
        else:
            dd = self.__cache['KMEANS'].predict(df[flds])

        for i in range(self.clusters):
            df[f'cluster_{i}'] = dd == i
            df[f'cluster_{i}'] = df[f'cluster_{i}'].astype(int)

        logging.debug('\t\tcompressing by PCA to %d features with %0.3f threshold',
                      self.compress, self.compress_threshold)
        if not self.__cache.get('PCA', None):
            flds.append('is_churned')
            target = df[flds].corr()['is_churned']
            self.__cache['PCA_fields'] = target[target.abs() < 0.05].abs().index.to_list()
            logging.debug('\t\t\tfound %d features to compress', len(self.__cache['PCA_fields']))
            self.__cache['PCA'] = PCA(n_components=self.compress, random_state=42)
            dd = self.__cache['PCA'].fit_transform(df[self.__cache['PCA_fields']])
        else:
            dd = self.__cache['PCA'].transform(df[self.__cache['PCA_fields']])

        for x in range(self.compress):
            df[f'pca_{x}'] = dd[:, x].astype(float)

    def transform(self, df):
        skp, exc = 0, []
        fields = sorted(set(df.columns.to_list()) - set(self.exclude))
        logging.info('Processing %d fields', len(fields))
        for col in fields:
            mt = sub(r'_(\d|less|more)+$', '', col.lower())
            if mt in self.exclude:
                skp += 1
                self.exclude.append(col)
                continue
            method = self.features.get(mt, getattr(self, '_' + mt, None))
            if callable(method):
                logging.debug('\t\t`%s` - SPECIAL METHOD', col)
                e = method(df, col)
                if type(e) == list:
                    exc += e
            else:
                logging.debug('\t\t`%s` - default outliner', col)
                self.__default_outliner(df, col)

        logging.info('\tprocessed %d, skipped %d', len(fields)-skp, skp)
        if self.fillNA is not None:
            logging.info('Fill the rest NA with `%s` value', self.fillNA)
            df.fillna(0, inplace=True)

        logging.info('Create new features')
        fields = sorted(set(fields) - set(self.exclude) - set(exc))
        self.createFeatures(df, fields)
