import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from easydict import EasyDict as edict
from loguru import logger

from tigren.datasets.graph import WikiGraphDataset
from tigren.datasets.stocks import NASDAQStockDataset
from tigren.datareaders import StockWindowDataReader
from tigren.losses import RankingMSELoss
from tigren.models import LSTMSequentialEmbedding, FCRelationalEmbedding, FCPrediction, TIGREN

def train():
    hyperparams = {
        "seq_embed_size": 64,
        "k_hops": 3,
        "hop_layers": 3,
        "lstm_layers": 3,
        "fn_layers": 3,
        "lr": 1e-5,
        "weight_decay": 1e-8,
        "alpha": 0.6,
        "window_size": 30,
        "epochs": 50,
        "batch_size": 4,
        "gradient_accumulation": 16
    }
    training_setup = {
        "ratios": (0.85, 0.15, 0.0),
        "date_range": ("2013-01-07", "2015-01-11"),
        "market": "NASDAQ",
        "wiki_date": "20180105",
        "device": "cuda:1",
        "output_path": "save/",
        "evaluation_steps": 32
    }
    options = {**hyperparams, **training_setup}
    options = edict(options)

    # 1. Load datasets
    nasdaq = NASDAQStockDataset(date_range=options.date_range)
    wikigraph = WikiGraphDataset(
        market=options.market, 
        filter_tickers=nasdaq.get_tickers(), 
        date=options.wiki_date
    )

    # 2. Process data & build readers that model.fit() can utilize
    train_df, valid_df = nasdaq.split_dataset(ratios=[0.8, 0.2])
    relational_encoding, binary_encoding = wikigraph.get_encodings()
    relational_encoding = torch.Tensor(relational_encoding).to(options.device)

    train_reader = StockWindowDataReader(stock_dataset=train_df, window_size=options.window_size, normalize=True)
    valid_reader = StockWindowDataReader(stock_dataset=valid_df, window_size=options.window_size, normalize=True)
    
    # 3. Define models
    input_data_shape, target_shape = train_reader.shape() # input_data_shape == (window_size, # of stocks, # of features per stock)

    sequential_embedding_model = LSTMSequentialEmbedding(input_data_shape, options.seq_embed_size, options.lstm_layers)
    relational_embedding_model = FCRelationalEmbedding(sequential_embedding_model.output_shape(), relational_encoding, options.k_hops, options.hop_layers)
    prediction_model = FCPrediction(sequential_embedding_model.output_shape(), relational_embedding_model.output_shape(), options.fn_layers)
    
    model = TIGREN(sequential_embedding_model, relational_embedding_model, prediction_model, device=options.device)

    # Define loss functions
    ranking_loss = RankingMSELoss(alpha=options.alpha)
    mse_loss = MSELoss()

    model.fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=ranking_loss,
        evaluation_metrics=[ranking_loss, mse_loss],
        batch_size=options.batch_size,
        epochs=options.epochs,
        lr=options.lr,
        weight_decay=options.weight_decay,
        output_path=options.output_path,
        evaluation_steps=options.evaluation_steps,
        save_best_model=True,
        show_progress_bar=False,
        gradient_accumulation=options.gradient_accumulation
    )



if __name__ == "__main__":
    train()
