import logging
from os import listdir
from os.path import isfile, join, exists
from datetime import datetime
from time import time
import pandas as pd

from .tools import loadDF, saveDF


class Extractor:
    """Extract train and test data from raw source CSV GZIPed files to dataaset"""

    def __init__(self, source: str = 'source', dataset: str = 'dataset'):
        """Init extractor with specific parameters

        Arguments:
            source {str} -- Folder of source files `source` by default
            dataset {str} -- Folder of result dataset `dataset` by default

        Raises:
            FileNotFoundError -- if train/test folder or sample.csv.gz not founds
        """
        self.sample_file = 'sample.csv.gz'
        self.train = source + '/train'
        self.test = source + '/test'
        self.dataset_train = dataset + '/train.csv.gz'
        self.dataset_test = dataset + '/test.csv.gz'

        if not exists(self.train):
            raise FileNotFoundError(f'Train folder `{self.train}` does not exists')
        if not exists(self.test):
            raise FileNotFoundError(f'Test folder `{self.test}` does not exists')
        if exists(self.dataset_train):
            logging.warning('Train dataset file `%s` already exists, will be overwritten', self.dataset_train)
        if exists(self.dataset_test):
            logging.warning('Test dataset file `%s` already exists, will be overwritten', self.dataset_test)

        self.files = [f for f in listdir(self.train) if isfile(join(self.train, f))]
        if self.sample_file not in self.files:
            logging.error('File `%s` not found', self.sample_file)
            raise FileNotFoundError(self.sample_file)
        self.files.remove(self.sample_file)
        logging.debug('... found these files to process:\n\t%s', self.files)

    def run(self, days: int = 7, intervals: int = 4, addMore: bool = True, addLess: bool = True):
        """Run extraction process and save datasets

        Keyword Arguments:
            days {int} -- Count days in interval (default: {7})
            intervals {int} -- Count intervals (default: {4})
            addMore {bool} -- Add param with prefix `_more` for data that more than higher interval (default: {True})
            addLess {bool} -- Add param with prefix `_less` for data that less than lower interval (default: {True})
        """
        ints = [(i+1, i+days) for i in range(0, days*intervals, days)]
        logging.info('Start extracting data from files, based on %s intervals %s %s', ints,
                     'and add `_more` param' if addMore else '',
                     'and add `_less` param' if addLess else '')

        start = time()
        self.__build('train', self.train, self.dataset_train, ints, addMore, addLess)
        logging.info('TRAIN dataset proccesing done in %0.2f sec', time()-start)

        start = time()
        self.__build('test', self.test, self.dataset_test, ints, addMore, addLess)
        logging.info('TEST dataset proccesing done in %0.2f sec', time()-start)

    def __build(self, name, source, destination, intervals, addMore, addLess):
        flds = ['user_id', 'is_churned', 'level', 'donate_total']
        rmset = set(['user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn'])

        def dlog(msg, *args):
            logging.debug('\t[%s] ' + msg, name, *args)

        def addFeature(df, flt, data, feature, fname):
            return pd.merge(df, data.loc[flt].
                            groupby('user_id')[feature].mean().reset_index().
                            rename(index=str, columns={feature: f'{feature}_{fname}'}),
                            how='left', on='user_id')

        dlog('Loading sample file')
        sample = loadDF(source + '/' + self.sample_file)
        if 'is_churned' not in sample.columns:
            flds.remove('is_churned')
        dataset = sample.copy()[flds]

        for f in self.files:
            start = time()
            dlog('Processing data from `%s` dataset', f)
            df = loadDF(source + '/' + f)
            if 'log_dt' not in df.columns:
                dlog('... STATIC data adding right to dataset, added columns %s', df.columns.drop('user_id').to_list())
                dataset = pd.merge(dataset, df, how='left', on='user_id')
            else:
                dlog('... DYNAMIC data calculate days num before churn')
                data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
                data['day_num_before_churn'] = 1 + (data['login_last_dt'] - data['log_dt']).apply(lambda x: x.days)
                df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

                features = list(set(data.columns) - rmset)
                dlog('...... processing found %s features', features)
                for feature in features:
                    if addLess:
                        dlog('......... adding %s less than inteval', feature)
                        df_features = addFeature(df_features, data['day_num_before_churn'] < intervals[0][0],
                                                 data, feature, 'less')
                    for i, inter in enumerate(intervals):
                        dlog('......... adding %s interval: %s', feature, inter)
                        df_features = addFeature(df_features, data['day_num_before_churn'].between(
                            inter[0], inter[1], inclusive=True), data, feature, i+1)
                    if addMore:
                        dlog('......... adding %s more than inteval', feature)
                        df_features = addFeature(df_features, data['day_num_before_churn'] > intervals[-1][1],
                                                 data, feature, 'more')

                dlog('... merge dynamic data to dataset')
                dataset = pd.merge(dataset, df_features, how='left', on='user_id')

            dlog('`%s` is done in %0.2f sec\n', f, time()-start)

        dlog('Saving result dataset')
        saveDF(dataset, destination)
