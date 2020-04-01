import logging
from os import listdir
from os.path import isfile, join
from datetime import datetime
from time import time
import pandas as pd

from .tools import loadDF


class Extractor:
    """Extract data from raw source CSV GZIPed files to dataaset"""

    def __init__(self, **config):
        """Init extractor with specific parameters

        Arguments:
            folder {str} -- Folder of source files, current by default.
            sample_file {str} -- Name of the sample file (main file of data), 'sample.csv.gz' by default
            output_file {str} -- Name of result file that will be created
            period {tuple, None} -- Should be tuple of (date from, date to) or None (default)
                to get from/to from sample file
            intervals {list[tuples], int}: Should be list of tuples (from, to) or int (default is 4) count of intervals
                that will be calculated from period

        Raises:
            FileNotFoundError -- if `sample_file` is not found in `folder`
        """
        self.folder = config.get('folder', '.')
        self.sample_file = config.get('sample_file', 'sample.csv.gz')
        self.output_file = config.get('output_file', 'dataset.csv.gz')
        self.period = config.get('period', None)
        self.intervals = config.get('intervals', 4)
        self.files = [f for f in listdir(self.folder) if isfile(join(self.folder, f))]
        logging.debug('In `%s` folder found these files:\n\t%s', self.folder, self.files)
        if self.sample_file not in self.files:
            logging.error('File `%s` not found', self.sample_file)
            raise FileNotFoundError(self.sample_file)
        else:
            self.files.remove(self.sample_file)

    def __loadDF(self, file):
        return loadDF(f'{self.folder}/{file}')

    def build(self):
        """Build and save gzipped dataset
        """
        logging.info('Read sample file')
        sample = self.__loadDF(self.sample_file)
        flds = ['user_id', 'is_churned', 'level', 'donate_total']
        if 'is_churned' not in sample.columns:
            flds.remove('is_churned')
        dataset = sample.copy()[flds]
        if not self.period:
            self.period = (sample['login_last_dt'].min(), sample['login_last_dt'].max())
        if type(self.intervals) == int:
            days = ((self.period[1]-self.period[0]).days + 1) // self.intervals
            self.intervals = [(i+1, i+1+days) for i in range(0, days*self.intervals, days+1)]
        logging.info('Extract data from files and attach to dataset,'
                     ' create features based on %s intervals', self.intervals)

        rmset = set(['user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn'])
        for f in self.files:
            start = time()
            logging.info('Processing data from `%s` dataset', f)
            df = self.__loadDF(f)
            if 'log_dt' not in df.columns:
                logging.debug('... static data adding right to dataset, added columns %s',
                              df.columns.drop('user_id').to_list())
                dataset = pd.merge(dataset, df, how='left', on='user_id')
            else:
                logging.debug('... dynamic data calculate features')

                data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
                data['day_num_before_churn'] = 1 + (data['login_last_dt'] -
                                                    data['log_dt']).apply(lambda x: x.days)
                df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

                features = list(set(data.columns) - rmset)
                logging.debug('...... processing found %s features', features)

                for feature in features:
                    for i, inter in enumerate(self.intervals):
                        flt = data['day_num_before_churn'].between(inter[0], inter[1], inclusive=True)
                        inter_df = data.loc[flt].\
                            groupby('user_id')[feature].mean().reset_index().\
                            rename(index=str, columns={feature: f'{feature}_{i+1}'})

                        df_features = pd.merge(df_features, inter_df, how='left', on='user_id')
                logging.debug('... add dynamic data')
                dataset = pd.merge(dataset, df_features, how='left', on='user_id')

            logging.info('Done in %0.2f sec', time()-start)

        logging.info('Saving dataset to `%s` file', self.output_file)
        dataset.to_csv(self.output_file, sep=';', index=False, compression='gzip')
        return dataset
