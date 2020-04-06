import logging
from datetime import datetime
from time import time
import pandas as pd


def loadDF(fileName):
    """Load file and parse it to dataframe, converts date fields to datetime

    Arguments:
        file {str} -- file to load

    Returns:
        DataFrame -- parsed dataframe
    """
    st = time()
    df = pd.read_csv(fileName, sep=';', na_values=['\\N', 'None'],
                     encoding='utf-8', compression='gzip')
    for i in ['login_last_dt', 'log_dt']:
        if i in df.columns:
            df[i] = df[i].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    logging.debug('\t\tfile `%s` loaded in %0.2f sec', fileName, time()-st)
    return df


def saveDF(df, fileName):
    """Save dataframe
    """
    st = time()
    df.to_csv(fileName, sep=';', index=False, compression='gzip')
    logging.debug('\t\tfile `%s` saved in %0.2f sec', fileName, time()-st)
    return True
