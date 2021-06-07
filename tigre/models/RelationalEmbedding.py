import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from loguru import logger

class RelationalEmbedding(nn.Module, ABC):
    def __init__(self,  seq_output_shape: Tuple[int, int], relational_encoding: np.ndarray, k_hops: int):
        """Defines the RelationalEmbedding model superclass. All RelationalEmbedding 
        models will inherit this class. This allows for standardization in defining a "relational embedding model".
        
        Args:
            seq_output_shape (Tuple[int, int, int]): The expected output shape of the sequential embedding model,
                whose output is the input to the relational embedding model.
            relational_encoding (np.ndarray): The graph definition. Expects shape: (N, N, K), 
                where K = number of possible relationships a given pair of nodes can have.
            k_hops (int): Number of hops to take in a graph for each node. If k_hops = 2, a computation graph
                for a node will consider not only the neighbors to the node but also the neighbors of those neighbors.
        """
        super(RelationalEmbedding, self).__init__()
        self.seq_output_shape = seq_output_shape
        self.relational_encoding = relational_encoding

        if type(self.relational_encoding) == list:
            self.relational_encoding = np.array(self.relational_encoding)
        
        relational_mask = relational_encoding.sum(dim=-1)
        relational_mask = torch.where(relational_mask > 0, 1, 0)
        self.relational_mask = relational_mask
        
        self.k_hops = k_hops
    
    @abstractmethod
    def config(self):
        """Outputs a config dictionary that contains all the necessary information for
        loading the model from a checkpoint.
        """
        pass
    
    @abstractmethod
    def forward(self, seq_embed):
        pass

    def output_shape(self) -> Tuple[int, int]:
        """Auto-compute the output shape of the RelationalEmbedding model. This will smooth out the 
        integration with the Prediction model. Recall that the embedding size of the relational embedding
        is the same size as the sequential embedding size.

        Returns:
            shape (Tuple[int, int]): Dimensions of one row output to the model.
                convention: (N (# of stocks), embedding_size)
        """
        N, seq_embed_size = self.seq_output_shape
        
        return (N, seq_embed_size)

class FCRelationalEmbedding(RelationalEmbedding):
    def __init__(
        self, 
        seq_output_shape: Tuple[int, int], 
        relational_encoding: np.ndarray, 
        k_hops: int,
        hop_layers: int
    ):
        """Fully-Conencted Relational Embedding model. This model uses Fully Connected NNs to 
        compute the relational embedding.

        Args:
            seq_output_shape (Tuple[int, int, int]): The expected output shape of the sequential embedding model,
                whose output is the input to the relational embedding model.
            relational_encoding (np.ndarray): The graph definition. Expects shape: (N, N, K), 
                where K = number of possible relationships a given pair of nodes can have.
            k_hops (int): Number of hops to take in a graph for each node. If k_hops = 2, a computation graph
                for a node will consider not only the neighbors to the node but also the neighbors of those neighbors.
            hop_layers (int): Number of Fully Connected layers for each hop. 
        """
        super().__init__(seq_output_shape, relational_encoding, k_hops)

        self.k_hops = k_hops
        self.hop_layers = hop_layers

        self.softmax_dim_1 = torch.nn.Softmax(dim=1)
        self.activation_function = nn.LeakyReLU()

        # Dummy checks
        if self.hop_layers <= 0:
            loger.error("hop_layers must be >= 1.")
            raise Exception

        # Making FC layers: linear_k_hop_layer_i
        U = seq_output_shape[-1] # U = sequential embedding size == relational embedding size
        K = relational_encoding.size(-1) # K = number of possible relationships a pair of nodes can take in the graph
        input_size = U + U + K # U + U + K
        for k in range(1,k_hops+1):
            for one_layer_i in range(1, hop_layers+1):
                if one_layer_i != hop_layers:
                    one_layer = nn.Linear(input_size, input_size)
                else:
                    one_layer = nn.Linear(input_size, 1)
                setattr(self, f"linear_{k}_hop_layer_{one_layer_i}", one_layer)

    def forward(self, seq_embed):
        """
        Parameters:
            seq_embed (Tensor size=(N, U)): sequential embedding matrix. N = len(stocks), U = seq embedding size. 
        Returns:
            relational_embeddings (Tensor size=(N, U))

        Felt reluctant about single-letter variable names but code became much more readable like this,
        especially when these are widely-agreed upon names.
        N = # of stocks
        U = sequential embedding size
        K = # of relation that pair a given two nodes

        The following is a walkthrough of one of the more confusing part of the code:

        Suppose seq_embed = [1,2,3] 

        seq_repeated = 
        | 1 1 1 |
        | 2 2 2 |
        | 3 3 3 |

        seq_repeated.transpose(0,1)=
        | 1 2 3 |
        | 1 2 3 |
        | 1 2 3 |

        seq_combined=
        | [1,1] [1,2] [1,3] |
        | [2,1] [2,2] [2,3] |
        | [3,1] [3,2] [3,3] |

        hence, seq_combined[i][j] = [e_i, e_j]
        where e_k = sequential embedding of kth stock.
        """
        batch_size, N, U = seq_embed.size()
        K = self.relational_encoding.size(-1)  # multi-hot binary graph encoding size.
        for k in range(1,self.k_hops+1):            
            # [N x U] => [(N x N) x U] => [N x N x U] => [N x N x (U + U)]
            encoding_repeated = self.relational_encoding.unsqueeze(dim=0).expand(batch_size, -1, -1, -1) # make it into a batch of size 1
            seq_repeated = seq_embed.repeat_interleave(repeats=N, dim=1)
            seq_repeated = seq_repeated.reshape((batch_size, N, N, U)) 
            seq_combined = torch.cat((
                seq_repeated,                   
                seq_repeated.transpose(1,2),
            ), dim=-1)
            # combined[i][j] = [e_i, e_j, a_ij]
            combined = torch.cat((
                seq_combined,
                encoding_repeated
            ), dim=-1)
            assert combined.size() == (batch_size, N, N, U+U+K)

            # set unconnected nodes to zero.
            mask_dim_expand = self.relational_mask.unsqueeze(dim=-1)
            combined = mask_dim_expand.mul(combined)
            
            # weights[i][j] = g(e_i, e_j, a_ij) values from Temporal paper.
            weights = combined
            for one_layer_k in range(1, self.hop_layers+1):
                linear_layer = getattr(self, f"linear_{k}_hop_layer_{one_layer_k}")
                weights = linear_layer(weights)
                weights = self.activation_function(weights)
            assert weights.size() == (batch_size, N, N, 1)

            # mask out disconnected nodes again
            weights = mask_dim_expand.mul(weights)
            weights = weights.squeeze()
            assert weights.size() == (batch_size, N, N)
            
            # refer to Temporal paper page 9. d_j = number of nodes satisfying sum(a_ij) > 0
            D = self.relational_mask.sum(dim=-1)
            scaled_weights = weights / D.unsqueeze(dim=-1)
            scaled_weights = scaled_weights.unsqueeze(dim=-1)
            assert scaled_weights.size() == (batch_size, N, N, 1)
            
            # weighted_embeds[i][j] = weight_j * [e_i, e_j]
            weighted_embeds = scaled_weights.mul(seq_combined)
            assert weighted_embeds.size() == (batch_size, N, N, U+U)

            # weighted_neigh_embeds[i][j] = weight_j * e_j
            weighted_neigh_embeds = weighted_embeds.split(U, dim=-1)[1]
            assert weighted_neigh_embeds.size() == (batch_size, N, N, U)

            # relational_embed[i] = weight_x * e_x + weight_y* e_y 
            # where weight_x + weight_y = 1
            seq_embed = torch.sum(weighted_neigh_embeds, dim=1)
            assert seq_embed.size() == (batch_size, N, U)
        return seq_embed

    def config(self):
        """Outputs a config dictionary that contains all the necessary information for
        loading the model from a checkpoint.
        """
        return {
            "output_shape": self.output_shape(),
            "seq_output_shape": self.seq_output_shape,
            "k_hops": self.k_hops,
            "hop_layers": self.hop_layers,
            "relational_encoding": self.relational_encoding.tolist(),
            "large_data": ["relational_encoding"]
        }
