from abc import abstractmethod, ABC

class GraphDataset(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_binary_encoding(self):
        pass

    @abstractmethod
    def get_relational_encoding(self):
        pass

    @abstractmethod
    def get_tickers(self):
        pass
    
    
