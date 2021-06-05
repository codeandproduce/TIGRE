import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .DataReader import DataReader

class StockWindowDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        super(StockWindowDataset, self)
        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)
    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx]
        }
    def __len__(self):
        return len(self.inputs)


class StockWindowDataReader(DataReader):
    def __init__(self, stock_dataset: pd.DataFrame, window_size: int, normalize=True):
        """If Dataset (e.g. NASDAQDataset) class is for conveniently loading raw data, a reader handles all the
        processing of the data and allows access to a torch DataLoader as well as the original dataset.
        
        Args:
            stock_dataset (pd.DataFrame): DataFrame object with type(index)=datetime.date and columns=[list of tickers].
            
        """
        self.normalize = normalize
        self.normalized_stock_dataset = self._normalize_data(stock_dataset=stock_dataset) # 1. Normalize data
        targets = self._build_ranking_targets(window_size=window_size, stock_dataset=self.normalized_stock_dataset) # 2. Build targets with ranking
        inputs = self._build_window_inputs(window_size=window_size, stock_dataset=self.normalized_stock_dataset)
    
        # prove 1 - 1 correspondence of price_window <=> pct change day after
        assert targets.shape[0] == inputs.shape[0] # number of rows
        assert targets.shape[1] == inputs.shape[2] # number of stocks. Recall that inputs.shape == [#rows, window_size, #stocks, #features]

        self.window_size = window_size
        self.inputs = inputs
        self.targets = targets
        self.dataset = StockWindowDataset(inputs, targets)
        
    def shape(self):
        return (self.inputs[0].shape, self.targets[0].shape)

    def get_dataset(self):
        return self.dataset

    def get_dataloader(self, batch_size: int, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        '''Dynamically create a PyTorch dataloader.

        Args:
            batch_size (int)
            shuffle (bool)
            drop_last (bool)
        Returns:
            dataloader (torch.utils.data.DataLoader)
        '''
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        return dataloader

    def _normalize_data(self, stock_dataset: pd.DataFrame) -> pd.DataFrame:
        """Normalize input data features.

        Args:
            stock_dataset (pd.DataFrame): Stock data with date as index and stocks as columns.
        Returns:
            normalized_stock_data (pd.DataFrame): Same dataset, except normalized

        @TODO Consider adding options to choose one of the many ways to normalize a dataset. 
        """
        
        min_val = stock_dataset.values.min()
        max_val = stock_dataset.values.max()
        normalized_stock_data = stock_dataset.apply(lambda x: (x - min_val) / (max_val - min_val))
        return normalized_stock_data

    def _build_ranking_targets(self, window_size: int, stock_dataset: pd.DataFrame) -> np.ndarray:
        """Targets for TIGREN model is the future performance ranking of stocks.

        Args:
            window_size (int): This is important in determining which of the pct_changes to use as targets.
                The target to the first input window of stocks is the pct change observed the day after.
            stock_dataset (pd.DataFrame): Stock data with date as index and stocks as columns
        Returns:
            pct_matrix (np.ndarray): Each row of pct_matrix is a list of pct_change (percent change) in price for each stock.
                Each row corresponds to a trading window of N days before.
        """
        pct_matrix = stock_dataset.pct_change()
        pct_matrix = pct_matrix.iloc[window_size + 1:] # refer to docstring about window_size

        pct_matrix = pct_matrix.applymap(lambda x: 0.01 if x == np.inf else x)

        pct_matrix = pct_matrix.values

        

        if self.normalize:
            max_val = pct_matrix.max()
            min_val = pct_matrix.min()
        
            pct_matrix = (pct_matrix - min_val) / (max_val - min_val)

        return pct_matrix
        
    def _build_window_inputs(self, window_size: int, stock_dataset: pd.DataFrame) -> np.ndarray:
        """Input to TIGREN is "windows" of past stock data. This function takes the historical stock data
        and builds the sliding window input data.

        Args:
            window_size (int): Window size to look back in the model
            stock_dataset (pd.DataFrame)
        Returns:
            inputs (np.ndarray): An array of stock trading windows.

        @TODO
        Currently, the dataset only has one feature: price. But we've built the model architecture
        so that it handles an array of features, since in the future we want it to be able to process 
        more features than just historical price. Currently each element in the stock_dataset dataframe
        is just the price value. Since the model expects a feature array, I've made an artificial feature
        array of [price]. That's the reason why we have inputs = np.expand_dims(inputs, axis=-1). This line of
        code should be deleted in the future once we have actual feature arrays.
        """
        n_windows = len(stock_dataset) - window_size - 1 # window count. -1 since we dont have next day pct_change target for the last window.
        n_stocks = len(stock_dataset.columns) # stocks count
        n_features = 1 # @TODO this needs to change once we can handle multiple features 
        
        inputs = [] # ws_idx = window start idx. Such that one_window = [ws_idx : ws_idx + window_size]
        for ws_idx in range(n_windows):
            window_X_data = stock_dataset.iloc[ws_idx : ws_idx+window_size].values
            assert window_X_data.shape == (window_size, n_stocks)

            inputs.append(window_X_data)

        inputs = np.array(inputs)
        assert inputs.shape == (n_windows, window_size, n_stocks)     

        inputs = np.expand_dims(inputs, axis=-1) # artifically make feature array. price -> [price]. Refer to @TODO.
        assert inputs.shape == (n_windows, window_size, n_stocks, n_features)   
    
        return inputs
    
    def __len__(self):
        return len(self.dataset)
    

    