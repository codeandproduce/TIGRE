import os
import pickle
import datetime
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from loguru import logger

class StockDataset(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def min(self):
        '''Minimum date in the dataset'''
        pass
    
    @abstractmethod
    def max(self):
        '''Maximum date in the dataset'''
        pass

    @abstractmethod
    def get_tickers(self):
        '''List of stock tickers in the dataset'''
        pass

    
    def _normalize_data(self, feature_data: dict) -> dict:
        """Normalize input data features.

        Args:
            feature_data (dict): feature_data.keys() == list of features. Representatively,
                feature_data["price"] is the pd.DataFrame of price data.
        Returns:
            normalized_stock_data (pd.DataFrame): Same dataset, except normalized

        @TODO Consider adding options to choose one of the many ways to normalize a dataset. 
        """
        self.min_vals = dict()
        self.max_vals = dict()
        
        for feature_name, df in feature_data.items():
            self.min_vals[feature_name] = df.values.min()
            self.max_vals[feature_name] = df.values.max()
            feature_data[feature_name] = df.apply(lambda x: (x - self.min_vals[feature_name]) / (self.max_vals[feature_name] - self.min_vals[feature_name]))
 

        return feature_data
    
    def denormalize(self, dataset, feature):
        """Denormalize data

        Flexible denormalization function. Since normalization parameters are separate for
        each feature, must specify which feature these are. 
        
        CASE 1. dataset == dict(), feature = None
            => dataset["price"] = pd.DataFrame of price data
        CASE 2. dataset == pd.DataFrame, feature = "price"
            => dataset == pd.DataFrame corresponding to the feature
        CASE 3. dataset == list of pd.DataFrames, feature = ["price", "feature1", ...]
            => dataset[i] == pd.DataFrame correspoding to feaure[i]


        CASE 4. dataset == 2D matrix of numbers
            CASE 4a: feature = "price" or some string. 
                => dataset == (window_size, # of stocks) for feature. 
            CASE 4b: feature = ["price", "feature1", ...]
                => dataset == (_, n_features)
                   dataset is a list of rows for each feature. i.e. dataset[i] = one row of values for feature[i]
        
        CASE 5. dataset == 3D matrix
            CASE 5a. feature = "price". 
                => dataset == (batch_size, window_size, # of stocks)
            CASE 5b. feature = ["price", "feature1", ...]
                => dataset == (window_size, n_stocks, # of features)
                OR
                => dataset == (window_size, # of features, n_stocks)
                OR
                => dataset == (# of features, window_size, n_stocks)

        CASE 6. dataset == 4D matrix
            => [batch_size, window_size, # of stocks, # of features] 
        """
        # CASE 1
        if type(dataset) == dict:
            return_dict = dict()

            if feature is not None:
                logger.warning("Is the request data a dictionary of feature dataframes? If so, feature parameter is unnecessary and is ignored.")
            else:
                for feature_name, feature_df in dataset.items:
                    denorm_func = self.get_denorm_func(feature_name)
                    return_dict[feature_name] = feature_df.apply(denorm_func)
                return return_dict

        # CASE 2
        if type(dataset) == pd.DataFrame:
            if type(feature) == list:
                if len(feature) > 1:
                    raise Exception("You cannot request one pd.DataFrame and request a list of features.")
                else:
                    feature = feature[0]
            denorm_func = self.get_denorm_func(feature_name)
            return dataset.apply(denorm_func)

        # CASE 3 
        if type(dataset) == list:
            if False not in [type(i) == pd.DataFrame for i in dataset]:
                if type(feature) is not list:
                    raise Exception("You can not request a list of dataframes and not request a list of corresponding features.")
                elif len(feature) != len(dataset):
                    raise Exception("Number of datasets requested does not match the number of requested features")

                return_list = []
                for feature_idx, one_feature in enumerate(feature):
                    denorm_func = self.get_denorm_func(feature_name)
                    feature_df = dataset[feature_idx]
                    feature_df = feature_df.apply(denorm_func)

                    return_list.append(feature_df)

                return return_list
            
        # Rest of the cases
        if type(dataset) == list or type(dataset) == np.array:
            if type(dataset) == list:
                dataset = np.array(dataset)

            if type(feature) == str:
                denorm_func = self.get_denorm_func(feature)
                return denorm_func(dataset)
            elif type(dataset) == list or type(dataset) == np.array:
                dataset_shape = list(dataset.shape)
                if len(dataset_shape) == 1: # one dimensional case
                    if len(feature) == len(dataset):
                        one_row = []
                        for feature_idx, feature_name in feature:
                            denorm_func = self.get_denorm_func(feature_name)
                            denorm_val = denorm_func(dataset[feature_idx])
                            one_row.append(denorm_val)
                    else:
                        raise Exception("Number of requested features does not match the requested data dimensions")
                else: # n-dimensional case
                    if len(feature) not in dataset_shape:
                        raise Exception ("Requested number of features does not match any dimensions in input data.")
                    else:
                        # at this point, we know dataset is a nD matrix and one of the dimension is n_features
                        n_feature_idx = dataset_shape.index(len(feature)) # for example, let n_feature_idx == 2
                        
                        default = list(range(len(dataset_shape))) # [0, 1, 2, 3]
                        del default[n_feature_idx] # [0, 1, 3]
                        default = tuple([n_feature_idx] + default) # [2, 0, 1, 3]
                        transposed_dataset = np.transpose(dataset, default) # transpose dataset to [n_features, ..whatever dims]
                        assert transposed_dataset.shape[0] == len(features) # confirm above line of thought.

                        for feature_idx, feature_name in features:
                            one_matrix = transposed_dataset[feature_idx]
                            denorm_func = self.get_denorm_func(feature_name)
                            one_matrix = denorm_func(one_matrix)
                            transposed_dataset[feature_idx] = one_matrix
                        
                        default = list(range(1, len(dataset_shape))) # [1, 2, 3]
                        default = default[:n_feature_idx] + [0] + default[n_feature_idx:] # [1, 2, 0, 3]
                        transposed_dataset = np.transpose(transposed_dataset, default) # transpose back

                        return transposed_dataset
        raise Exception("Could not interpret this data and feature input.")
                            
                            
    
    def get_denorm_func(self, feature_name: str):
        if feature_name not in self.max_vals.keys() or feature_name not in self.min_vals.keys():
            raise Exception("Requested feature does not exist.")
        max_val = self.max_vals[feature_name]
        min_val = self.min_vals[feature_name]
        denorm_func = lambda x: min_val + (x * (max_val - min_val))
        return denorm_func

    def _str_to_date(self, date_str):
        date_str = date_str.split("/")
        or_str = []
        for ss in date_str:
            or_str = or_str + ss.split("-")
        Y, m, d = or_str
        return datetime.date(int(Y), int(m), int(d))
    
    def split_dataset(self, ratios=List[int]) -> List[np.ndarray]:
        """Splits the dataset *in order* in above ratios. For instance if the data had 10 trading days
        and ratios=(0.6, 0.2, 0.2), this will split the data as {first 6 days, the next 2 days, the next 2 days}.

        Args:
            ratios (List[int]): variable length list of ratios to split the data. 
        
        Returns:
            splits (List[np.ndarray])
        """
        metadata = self.metadata()
        n_rows = metadata["n_rows"]
        date_index = metadata["date_index"]

        assert len(date_index) == len(self.data)

        for idx, ratio in enumerate(ratios):
            ratios[idx] = int(n_rows*ratio)

        splits = []
        prev_index = 0
        for idx, ratio in enumerate(ratios):
            data_split = self.data[prev_index : prev_index + ratio]
            logger.info(f"datasplit {idx} start={date_index[prev_index]} end={date_index[prev_index + ratio-1]} count={len(data_split)}")
            splits.append(data_split)
            prev_index = prev_index + ratio

        return splits
        
    def split_train_valid_test_random(self, ratios: tuple, random_state: int = 100):
        """Randomly generate split: train - valid - test. Ratios are supplied in that order.
        This is probably not useful for stock data, but included nevertheless for now. 
        Args:
            ratios       (tuple): ratio of train - valid - test, respectively.
            random_state (int): random seed for reproducibility
        Returns:
            train (pd.DataFrame): DataFrame object of the train data
            valid (pd.DataFrame)
            test  (pd.DataFrame)

        """
        summed = sum(ratios)
        if summed != 1.0:
            raise Exception("Ratios must add up to 1.0. EX: (0.6, 0.2, 0.2)")
        else:
            cache_exists = False
            if self.cache: # you want there to be a cache
                la = ["train", "valid", "test"]
                filenames = [f"cache/raw_{i}_split.pkl" for i in la]
                cache_exists = [os.path.isfile(fn) for fn in filenames]
                cache_exists = not (False in cache_exists)
                if cache_exists: # and you had the cache!
                    train = pickle.load(open(f"cache/raw_train_split.pkl", "rb"))
                    valid = pickle.load(open(f"cache/raw_valid_split.pkl", "rb"))
                    test = pickle.load(open(f"cache/raw_test_split.pkl", "rb"))
                    return train, valid, test

            if not cache_exists: # there is no cache 
                df = self.data
                train = df.sample(frac=ratios[0], random_state=random_state)
                rest = df.drop(train.index)
                valid_frac = ratios[1] / (ratios[1] + ratios[2])
                valid = rest.sample(frac=valid_frac, random_state=random_state)
                test = rest.drop(valid.index)        

                if self.cache: # but you wanted there to be cache. so make cache.
                    logger.warning("Cached train-valid-test split not found. Generating new split but <r>do reconfirm this is the intended consequence</r>.")
                    ll = [train, valid, test]
                    zipped = list(zip(ll, la))
                    for zipp in zipped:
                        with open(f"cache/raw_{zipp[1]}_split.pkl", "wb") as f:
                            pickle.dump(zipp[0], f)

                return train, valid, test

    