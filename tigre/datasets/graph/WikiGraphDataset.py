import os
import numpy as np
import pandas as pd
import torch
from loguru import logger
from typing import Tuple, List
from tigre.datasets.graph.GraphDataset import GraphDataset

def get_market_npy_url(date, market):
    return f"tigre/datasets/data/wiki/{date}/{market.upper()}_wiki_relation.npy"
def get_ticker_csv_url(date, market):
    return f"tigre/datasets/data/wiki/{date}/{market.upper()}_tickers.csv"

class WikiGraphDataset(GraphDataset):
    def __init__(self, market: str, filter_tickers: List[str]=None, date: str="20180105"):
        """Loads wiki graph. This will allow retrieval of the adjacency matrix and the 
        full relational encoding. Recall that a relational encoding is a matrix that shows not only
        if two stocks are connect but also the relationship that defines the relationship.

        Args:
            market (str): "NASDAQ" or "NYSE"
            filter_tickers (List[str]): We might want the graph of only a certain portion
                of the market. filter_tickers is the list of stocks that we want a graph of.
            dump_date (str): Which day's wikidata dump we should use.

        @TODO Figure out a way to dynamically load wikidata. Currently we can only provide 20180105
        since we've having trouble processing such a massive load.  
        """
        wiki_data = self.load_local_wiki(date, market) 
        if wiki_data is None: # data does not exist locally. Refer to @TODO
            raise Exception("Module currently does not support dynamically pulling wiki graphs from the web.")
        else:
            self.relational_encoding, self.tickers = wiki_data
        
        if filter_tickers:
            self.relational_encoding = self._filter_encoding_tickers(
                relational_encoding=self.relational_encoding,
                current_tickers=self.tickers,
                filter_tickers=filter_tickers
            )
            self.tickers = filter_tickers

        self.binary_encoding = self.relational_to_binary(self.relational_encoding)

        self.relational_encoding = torch.Tensor(self.relational_encoding)
        self.binary_encoding = torch.Tensor(self.binary_encoding)

    def load_local_wiki(self, date: str, market: str) -> Tuple[np.ndarray, List[str]]:
        """Loads locally stored wiki graph for corresponding date & market.

        Args:
            date (str)
            market (str)
        Returns:
            relational_encoding (np.ndarray): Matrix representation of a graph where
                each element is a vector of length = # of possible relationships. An enhanced
                idea of a traditional adjacency graph where instead of just binary indicators of
                whether two nodes are connected, it shows a vector that indicates which, among the
                many relationships two nodes can have, connects the two.
            tickers (List[str]): Corresponding list of tickers for the encoding. 
                i.e. relational_encoding[i][j] == relationship of stock_i and stock_j
                    where tickers[i] == stock_i, tickers[j] == stock_j
        """
        if os.path.isdir(f"tigre/datasets/data/wiki/{date}"):
            market_npy_url = get_market_npy_url(date, market)
            ticker_csv_url = get_ticker_csv_url(date, market)


            relation_encoding = np.load(market_npy_url)
            universe_tickers = pd.read_csv(ticker_csv_url, header=None)
            universe_tickers = universe_tickers.iloc[:,0].tolist()
            return relation_encoding, universe_tickers
        else:
            logger.warning(f"Could not find local Wiki graph for date={date}, market={market}.")
            return None
    
    def relational_to_binary(self, relational_encoding: np.ndarray) -> np.ndarray:
        """Given a relational_encoding, generate a traditional binary adjacency matrix.

        Args:
            relational_encoding (np.ndarray): Concept explained in load_local_wiki(date, market).
        Returns:
            binary_encoding (np.ndarray): Traditional binary adjcacency matrix where we have values 1
                for two notes that are connected and 0s for unconnected nodes. 
        """
        rel_shape = [relational_encoding.shape[0], relational_encoding.shape[1]]
        mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                            np.sum(relational_encoding, axis=2))
        mask = np.where(mask_flags, np.zeros(rel_shape, dtype=int), np.ones(rel_shape, dtype=int))
        return mask
    
    def _filter_encoding_tickers(self, relational_encoding: np.ndarray, current_tickers: List[str], filter_tickers: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Filters out unwanted tickers from our graph. 
        We require that filter_tickers \subset current_tickers.

        Args:
            relational_encoding (np.ndarray): Current relational encoding graph
            current_tickers (List[str]): Current list of tickers.
            filter_tickers (List[str]): List of desired tickers that we want the graph for.
        Returns: 
            relational_encoding (np.ndarray): New, filtered relational encoding.
        """
        # Check whether every ticker is in the market
        complements = [i for i in filter_tickers if i not in current_tickers]
        assert len(complements) == 0, "filter_ticker contains tickers non-existent in the original graph."

        # Index of desired tickers in the original graph.
        ticker_idx_list = []
        for ticker in filter_tickers:
            ticker_idx_list.append(current_tickers.index(ticker))
            
        # Make a new relation encoding with the desired tickers only
        relational_encoding = np.take(relational_encoding, ticker_idx_list, axis=0)
        relational_encoding = np.take(relational_encoding, ticker_idx_list, axis=1)
    
        # Check for proper slicing.
        N = len(filter_tickers) # number of desired tickers
        R = len(relational_encoding[0][0]) # number of possible relationships
        assert relational_encoding.shape == (N, N, R)

        return relational_encoding

    def get_encodings(self):
        return self.relational_encoding, self.binary_encoding

    def get_relational_encoding(self):
        return self.relational_encoding

    def get_binary_encoding(self):
        return self.binary_encoding
        
    def get_tickers(self):
        return self.tickers
        


