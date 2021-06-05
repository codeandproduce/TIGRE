import datetime
import pickle
import pandas as pd
from typing import List, Tuple
from loguru import logger

from .StockDataset import StockDataset

NASDAQ_PICKLE_URL = 'tigren/datasets/data/NASDAQ_price.pkl'

class NASDAQStockDataset(StockDataset):
    def __init__(self, date_range: Tuple[str, str] = (None, None), features: List[str] = None):
        """Loads the NASDAQ dataset.
        
        Args:
            date_range (Tuple[str, str]): Only retrieve a certain time range of the market. This is an inclusive range, i.e. 
                both the starting and ending index are included in the range. If = (None, None), returns the full dataset.
                Example: ("2015-01-03", "2016-01-03"). 
                Flexible string format. e.g. "2015/1/05" or even "2015-1/05"
                Min date and max date can be found in min(), max().
            features (List[str]): List of features to load from data. 
            
        @TODO
        Currently, we only have the pickled dataset for price features. Change Dataset
        code such that we can produce different datasets for different requested list of features. 
        Also refer to docstrings in tigren/datareaders/StockWindowDataReader.py > _build_window_inputs
        """
        self.features = features

        with open(NASDAQ_PICKLE_URL, "rb") as f:
            data = pickle.load(f) # load data

            # default date range: full date range.
            self.start_date = data.index[0]
            self.end_date = data.index[-1]

            # adjust to requested date range
            if date_range[0] is not None:
                self.start_date = self._str_to_date(date_range[0])
            if date_range[1] is not None:
                self.end_date = self._str_to_date(date_range[1])
            date_range = pd.date_range(self.start_date, self.end_date)

            self.data = data.loc[list(data.index.intersection(date_range))] # necessary since there's no data for weekends and holidays.
            self.tickers = self.data.columns.tolist()

        self.min_date = self.data.index[0].strftime("%Y-%m-%d")
        self.max_date = self.data.index[-1].strftime("%Y-%m-%d")
        
    def metadata(self):
        info = {
            "n_tickers": len(self.tickers),
            "n_features": 1 if self.features is None else len(self.features), # REFER TO @TODO
            "n_rows": len(self.data)
        }
        return info
    def max(self):
        return self.max_date

    def min(self):
        return self.min_date
        
    def get_tickers(self):
        return self.tickers
            
        
    