import datetime
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
from loguru import logger

from .StockDataset import StockDataset

# NASDAQ_PICKLE_URL = 'tigre/datasets/data/NASDAQ_price.pkl'

# @TODO in the future, change code so that these datasets are available on an external link
# and caches the dataset locally after being requested once. For now we just store the raw csv
# in the project directory.
NASDAQ_CSV = {
    "price": "tigre/datasets/data/NASDAQ_price.csv",
    "feature1": "tigre/datasets/data/NASDAQ_feature1.csv"
}

class NASDAQStockDataset(StockDataset):
    def __init__(self, date_range: Tuple[str, str] = (None, None), features: List[str] = ["price", "feature1"], normalize: bool = True):
        """Loads the NASDAQ dataset.
        
        Args:
            date_range (Tuple[str, str]): Only retrieve a certain time range of the market. This is an inclusive range, i.e. 
                both the starting and ending index are included in the range. If = (None, None), returns the full dataset.
                Example: ("2015-01-03", "2016-01-03"). 
                Flexible string format. e.g. "2015/1/05" or even "2015-1/05"
                Min date and max date can be found in min(), max().
            features (List[str]): List of features to load from data. 
            normalize (bool): If set to True, normalizes data and stores the min/max values used in normalization for
                denormalization purposes in the future.
            
        CHANGES:
        self.data is now a dictionary of dataframes. When previously it was just self.data = price data dataframe, now it's:
        self.data["price"] == price data dataframe
        self.data.keys() == list of features.

        I thought of other approaches like
        self.data.iloc[0]["AAPL"] = {"price": ..., "feature1": ...}
        but this is a way more efficient method.

        The code also forces that you must have at least the "price" feature. Other features are optional.
        """
        self.normalize = normalize
        self.features = features
        self.data_dict = dict()

        for feature_name in features:
            if feature_name in NASDAQ_CSV.keys():
                csv_path = NASDAQ_CSV[feature_name]
                df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
                self.data_dict[feature_name] = df
            else:
                raise Exception(f"Requested feature {feature_name} does not exist.")

        if "price" not in self.data_dict.keys():
            raise Exception(f"Price is a required feature.")

        price_df = self.data_dict["price"]
        self.tickers = price_df.columns.tolist()

        """
        1. Slice the data to the requested date range.
        """
        # default date range: full date range. Remember that index is list of dates
        start_date = price_df.index[0]
        end_date = price_df.index[-1]

        # adjust to requested date range
        if date_range[0] is not None:
            start_date = self._str_to_date(date_range[0])
        if date_range[1] is not None:
            end_date = self._str_to_date(date_range[0])
        self.date_range = pd.date_range(start_date, end_date)
    
        for data_name, df in self.data_dict.items():
            inter = list(df.index.intersection(self.date_range))
            self.data_dict[data_name] = df.loc[inter]
        self.min_date = self.data_dict["price"].index[0].strftime("%Y-%m-%d")
        self.max_date = self.data_dict["price"].index[-1].strftime("%Y-%m-%d")
        self.date_index = self.data_dict["price"].index

        if normalize:
            self.data_dict = self._normalize_data(self.data_dict)

        price_np = self.data_dict["price"].values
        ordered_np_list = [price_np] + [self.data_dict[key].values for key in self.features if key != "price"]
        ordered_np_list = [np.expand_dims(i, axis=-1) for i in ordered_np_list]

        self.data = np.concatenate(tuple(ordered_np_list), axis=-1)
        assert self.data.shape[-1] == len(features)

    def metadata(self):
        info = {
            "n_tickers": len(self.tickers),
            "n_features": 1 if self.features is None else len(self.features), # REFER TO @TODO
            "n_rows": len(self.data),
            "date_index": self.date_index
        }
        return info
    def max(self):
        return self.max_date

    def min(self):
        return self.min_date
        
    def get_tickers(self):
        return self.tickers
            
        
    