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
    def __init__(self, stock_dataset: np.ndarray, target_index:int, window_size: int):
        """If Dataset (e.g. NASDAQDataset) class is for conveniently loading raw data, a reader handles all the
        processing of the data and allows access to a torch DataLoader as well as the original dataset.
        
        Args:
            stock_dataset (np.ndarray): numpy ndarray with shape [# of dates, # of stocks, # of features]
            
        """
        self.stock_dataset = stock_dataset
        self.target_index = target_index
        targets = self._build_ranking_targets(window_size=window_size, stock_dataset=self.stock_dataset, target_index=self.target_index) # 2. Build targets with ranking
        inputs = self._build_window_inputs(window_size=window_size, stock_dataset=self.stock_dataset)
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

    def _build_ranking_targets(self, window_size: int, stock_dataset: np.ndarray, target_index: int) -> np.ndarray:
        """Targets for TIGREN model is the future performance ranking of stocks.

        Args:
            window_size (int): This is important in determining which of the pct_changes to use as targets.
                The target to the first input window of stocks is the pct change observed the day after.
            stock_dataset (pd.DataFrame): Stock data with date as index and stocks as columns
        Returns:
            pct_matrix (np.ndarray): Each row of pct_matrix is a list of pct_change (percent change) in price for each stock.
                Each row corresponds to a trading window of N days before.
        """
    
        target_np = stock_dataset[:, :, target_index] # [# of days, # of stocks, target_index of feature]
        pct_matrix = np.diff(target_np, axis=0) / target_np[:-1, :]
        pct_matrix = pct_matrix[window_size:] # refer to docstring about window_size
        temp_func = lambda x: 0.01 if x == np.inf else x
        vfunc = np.vectorize(temp_func)
        pct_matrix = vfunc(pct_matrix)

        return pct_matrix
        
        
    def _build_window_inputs(self, window_size: int, stock_dataset: np.ndarray) -> np.ndarray:
        """Input to TIGREN is "windows" of past stock data. This function takes the historical stock data
        and builds the sliding window input data.

        Args:
            window_size (int): Window size to look back in the model
            stock_dataset (np.ndarray)
        Returns:
            inputs (np.ndarray): An array of stock trading windows.

        """
        n_windows = len(stock_dataset) - window_size - 1 # window count. -1 since we dont have next day pct_change target for the last window.
        n_stocks = len(stock_dataset[0]) # stocks count
        n_features = len(stock_dataset[0][0])
        
        inputs = [] # ws_idx = window start idx. Such that one_window = [ws_idx : ws_idx + window_size]
        for ws_idx in range(n_windows):
            window_X_data = stock_dataset[ws_idx : ws_idx+window_size]
            assert window_X_data.shape == (window_size, n_stocks, n_features)
            inputs.append(window_X_data)

        inputs = np.array(inputs)
        assert inputs.shape == (n_windows, window_size, n_stocks, n_features)     
    
        return inputs
    
    def __len__(self):
        return len(self.dataset)
    

    