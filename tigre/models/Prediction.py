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
    def __init__(
        self, 
        seq_input_shape: Tuple[int, int], 
        rel_input_shape: Tuple[int, int], 
        n_layers: int,
        h_size: int,
        double: bool
    ):
        """Defines the Fully-Connected Prediction model.

        Args:
            seq_input_shape (Tuple[int, int]): The output shape of the sequential embedding model that is the
                input to this model.
            rel_input_shape (Tuple[int, int]): The output shape of the relational embedding model that is the
                input to this model.
            n_layers (int): Number of Fully-Connected layers.
        """
        super().__init__(seq_input_shape, rel_input_shape)

        input_size = 2*seq_input_shape[-1] # {seq_embed_size + rel_embed_size} == 2 * seq_embed_size
        output_size = 1 # one score per one stock 


        # if self.double:
        #     torch.set_default_dtype(torch.float64)
        # else:
        #     torch.set_default_dtype(torch.float32)

        self.input_size = input_size
        self.n_layers = n_layers

        self.norm_1 = nn.BatchNorm1d(seq_input_shape[0])
        self.fc_1 = nn.Linear(input_size, h_size, bias=True)
        self.af_1 = nn.ReLU()
        for n in range(2, self.n_layers):
            fc_layer_name = f"fc_{n}"
            af_name = f"af_{n}"
            norm= f"norm_{n}"
            setattr(self,norm, nn.BatchNorm1d(seq_input_shape[0]))
            setattr(self, fc_layer_name, nn.Linear(h_size, h_size, bias=True))
            setattr(self, af_name, nn.ReLU())
        self.norm_last = nn.BatchNorm1d(seq_input_shape[0])
        self.fc_last = nn.Linear(h_size, 1, bias=True)
        self.af_last = nn.ReLU()

    def forward(self, seq_embeddings, relational_embeddings):
        combined_embeddings = torch.cat((
            seq_embeddings,
            relational_embeddings
        ), dim=-1)
        
        x = combined_embeddings
        # if self.double:
        #     x = x.type(torch.float64)
        # else:
        #     x = x.type(torch.float32)

        for n in range(1, self.n_layers):
            fc_layer_name = f"fc_{n}"
            af_name = f"af_{n}"
            norm_name = f"norm_{n}"
            norm = getattr(self, norm_name)
            layer = getattr(self, fc_layer_name)
            af = getattr(self, af_name)
            x = norm(x)
            x = layer(x)
            x = af(x)
        x = self.norm_last(x)
        x = self.fc_last(x)
        x = self.af_last(x)
        x = torch.squeeze(x, dim=-1)
        return x

    def config(self):
        """Outputs a config dictionary that contains all the necessary information for
        loading the model from a checkpoint.

        Considered just automating this process with __dict__ and such but this gives us
        greater control over this process.
        """
        return {
            "seq_input_shape": self.seq_input_shape,
            "rel_input_shape": self.rel_input_shape,
            "n_layers": self.n_layers,
            "h_size": self.h_size,
            "double": self.double
        }
        pass
        
