import logging
from re import sub
from os.path import exists
from datetime import datetime
from time import time
import pandas as pd

from .tools import loadDF


class Preparer:
    """Remove outliners, fill null data from prepared dataaset"""

    def __init__(self, input_file: str, output_file: str):
        """Init preparer with specific parameters

        Arguments:
            input_file {str} -- FileName of the dataset file
            output_file {str} -- FileName of result file that will be created

        Raises:
            FileNotFoundError -- if `input_file` is not found
        """
        self.input_file = input_file
        self.output_file = output_file

        if not exists(self.input_file):
            logging.error('File `%s` not found', self.input_file)
            raise FileNotFoundError(self.input_file)

    def _age(self, dataset, col):
        median = round(dataset['age'].median())
        dataset['age'] = dataset['age'].fillna(median)
        dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = median

    def _gender(self, dataset, col):
        mode = dataset['gender'].mode()[0]
        dataset['gender'] = dataset['gender'].fillna(mode)
        dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = mode
        dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})

    def _days_between_fl_df(self, dataset, col):
        dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1

    def _avg_min_ping(self, dataset, col):
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()

    def _core_type_nm(self, dataset, col):
        for i in ['Core', 'Returns', 'Mantle']:
            dataset[f'Type_{i}'] = dataset[col] == i
            dataset[f'Type_{i}'] = dataset[f'Type_{i}'].astype(int)
        dataset.drop(col, axis=1, inplace=True)

    def build(self):
        """Build and save gzipped cleaned dataset
        """

        logging.info('Read sample file')
        df = loadDF(self.input_file)

        logging.info('Special processing')
        for col in df.columns:
            method = getattr(self, sub(r'_\d+$', '', '_' + col.lower()), None)
            if callable(method):
                logging.debug('... proccesing `%s` column', col)
                method(df, col)

        logging.info('Fill the rest')
        df.fillna(0, inplace=True)

        logging.info('Saving dataset to `%s` file', self.output_file)
        df.to_csv(self.output_file, sep=';', index=False, compression='gzip')
        return df
