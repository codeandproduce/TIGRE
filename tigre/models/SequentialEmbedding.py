from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from loguru import logger

class SequentialEmbedding(nn.Module, ABC):
    def __init__(self, input_shape: Tuple[int, int, int], embedding_size: int):
        """Defines the SequentialEmbedding model superclass. All SequentialEmbedding
        models will inherit this class.

        Args:
            input_shape (Tuple[int, int, int]): The dimensions of the data that will be input 
                to the model. This needs to follow the convention: (window_size, N (# of stocks), n_features)
            embedding_size (int): Size of the sequential embedding per stock. 
        """        
        super(SequentialEmbedding, self).__init__()
        self.input_shape = input_shape
        self.embedding_size = embedding_size

        window_size, N, n_features = input_shape
        self.window_size = window_size
        self.N = N
        self.n_features = n_features
        
    @abstractmethod
    def config(self):
        """Outputs a config dictionary that contains all the necessary information for
        loading the model from a checkpoint.
        """
        pass
    
    @abstractmethod
    def forward(self, input_data):
        pass

    
    def output_shape(self) -> Tuple[int, int]:
        """Auto-compute the output shape of the SequentialEmbedding model. This will smooth out the 
        integration with the Relational Embedding.

        Returns:
            shape (Tuple[int, int, int]): Dimensions of one row output to the model. 
                (window_size, N (# of stocks), embedding_size)
        """
        shape = (self.N, self.embedding_size)
        return shape

        
class LSTMSequentialEmbedding(SequentialEmbedding):
    def __init__(self, input_shape: Tuple[int, int, int], embedding_size:int, lstm_layers: int,  hidden_dim: int):
        """LSTM-based Sequental Embedding model.
        """
        super().__init__(input_shape, embedding_size)
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim

        self.lstm = torch.nn.RNN(
            input_size=self.n_features, 
            hidden_size=self.hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True,
            bias=True
        )
       
        self.fc = nn.Linear(self.hidden_dim, embedding_size, bias=True)

    def forward(self, input_data):
        '''
        Here was a source of a huge logic bug confusion.
        Say N = 736, batch_size = 16, window_size = 30.

        Then input_data.size() == (16, 30, 736) == (batch_size, seq, features???????)
        and out.size() became == (16, 64), suggesting:
            out[0] == "seq embedding" for (30, 736).

        which isn't what we want! if you think about it, its more like:
            we have 16 rows in a batch and 736 stocks in each row, and a single-valued feature
            for each stock.

        so we actually need to do
        (16, 30, 736) => (16 * 736, 30) => LSTM => (16 * 736, 64) => (16, 736, 64)

        and we actually should work with (16, 30, 736, 1) instead of (16, 30, 736) so that:
            1. we avoid future confusion by dividing dimensions with function:
                - being able to say: [batch_size, window_size, N, n_features]
            2. are we really only going to use one feature per stock???
        '''
        batch_size = input_data.size(0)
        window_size = input_data.size(1)
        n_stocks = input_data.size(2)
        n_features = input_data.size(3)

        assert self.window_size == window_size
        assert self.N == n_stocks
        assert self.n_features == n_features
        

        input_data = input_data.permute(0, 2, 1, 3) # (16, 30, 736, 1) => (16, 736, 30, 1)
        
        input_data = input_data.reshape([batch_size * self.N, self.window_size, self.n_features]) # => (16*736, 30, 1)
        out, (hn, cn)  = self.lstm(input_data) # (16*736, 30, 1) => (16 * 736, 30, 64)
        out = self.fc(out[:, -1, :])
        # out = out[:, -1, :] # (batch_size * n_stocks, embeddings per stock)
        out = out.reshape(batch_size, self.N, self.embedding_size) # => (batch_size, n_stocks, 64)
        return out # (batch_size, N, U)

    def config(self):
        """Outputs a config dictionary that contains all the necessary information for
        loading the model from a checkpoint.
        """
        return {
            "output_shape": self.output_shape(),
            "input_shape": self.input_shape,
            "embedding_size": self.embedding_size,
            "lstm_layers": self.lstm_layers,
            "hidden_dim": self.hidden_dim
        }
        pass