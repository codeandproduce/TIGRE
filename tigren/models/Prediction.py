from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Tuple, List

class Prediction(nn.Module, ABC):
    def __init__(self, seq_input_shape: Tuple[int, int], rel_input_shape: Tuple[int, int, int]):
        """Defines the Prediction model superclass. All Prediction models that will inherit this class.
        This model's inputs are the outputs of both the Sequential Embedding model and the Relational Embedding model
        to produce a final prediction.

        Args:
            seq_input_shape (Tuple[int, int]): The output shape of the sequential embedding model that is the
                input to this model.
            rel_input_shape (Tuple[int, int]): The output shape of the relational embedding model that is the
                input to this model.
        """
        super(Prediction, self).__init__()
        self.seq_input_shape = seq_input_shape
        self.rel_input_shape = rel_input_shape
    
    @abstractmethod
    def config(self):
        """Outputs a config dictionary that contains all the necessary information for
        loading the model from a checkpoint.
        """
        pass
    
class FCPrediction(Prediction):
    def __init__(self, seq_input_shape: Tuple[int, int], rel_input_shape: Tuple[int, int], layers: int):
        """Defines the Fully-Connected Prediction model.

        Args:
            seq_input_shape (Tuple[int, int]): The output shape of the sequential embedding model that is the
                input to this model.
            rel_input_shape (Tuple[int, int]): The output shape of the relational embedding model that is the
                input to this model.
            layers (int): Number of Fully-Connected layers.
        """
        super().__init__(seq_input_shape, rel_input_shape)

        input_size = 2*seq_input_shape[-1] # {seq_embed_size + rel_embed_size} == 2 * seq_embed_size
        output_size = 1 # one score per one stock 

        self.layers = layers
        self.activation_function = nn.LeakyReLU()
        for one_layer in range(1, layers+1):
            if one_layer != layers:
                setattr(self, f"linear_layer_{one_layer}", nn.Linear(input_size, input_size))
            else:
                setattr(self, f"linear_layer_{layers}", nn.Linear(input_size, output_size))

    def forward(self, seq_embeddings, relational_embeddings):
        combined_embeddings = torch.cat((
            seq_embeddings,
            relational_embeddings
        ), dim=-1)
        
        for one_layer in range(1, self.layers+1):
            linear_layer = getattr(self, f"linear_layer_{one_layer}")
            combined_embeddings = linear_layer(combined_embeddings)
            combined_embeddings = self.activation_function(combined_embeddings)
        return combined_embeddings

    def config(self):
        """Outputs a config dictionary that contains all the necessary information for
        loading the model from a checkpoint.

        Considered just automating this process with __dict__ and such but this gives us
        greater control over this process.
        """
        return {
            "seq_input_shape": self.seq_input_shape,
            "rel_input_shape": self.rel_input_shape,
            "layers": self.layers
        }
        pass
    
    

        