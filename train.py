import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from easydict import EasyDict as edict
from loguru import logger

from tigre.datasets.graph import WikiGraphDataset
from tigre.datasets.stocks import NASDAQStockDataset
from tigre.datareaders import StockWindowDataReader
from tigre.losses import RankingMSELoss
from tigre.models import LSTMSequentialEmbedding, FCRelationalEmbedding, FCPrediction
from tigre import TIGRE

def normalize_denormalize():
    """Code example for loading data & denormalizing for analysis purposes:
    """
    nasdaq = NASDAQStockDataset(date_range=(None, None), normalize=True)

    train, valid, test = nasdaq.split_dataset(ratios=[0.8, 0.1, 0.1])

    train_reader = StockWindowDataReader(stock_dataset=train, target_index=0, window_size=30)
    train_loader = train_reader.get_dataloader(batch_size=2, shuffle=False)

    # EXAMPLE 1. Denormalize a pd.DataFrame
    denormalized_1 = nasdaq.denormalize(train_df)
    
    # EXAMPLE 2. Denormalize during batch dataloading:
    for batch in train_loader:
        inputs = batch["inputs"]
        denormalized_2 = nasdaq.denormalize(inputs)
    
        # EXAMPLE 3. literally any kind of dumb use:
        nn = inputs[0].tolist()
        denormalized_3 = nasdaq.denormalize(nn)

        print(train_df.iloc[0].to_list()[0])
        print(denormalized_1.iloc[0].to_list()[0])
        print()
        print(inputs[0][0][0])
        print(denormalized_2[0][0][0])
        print()
        print(nn[0][0][0])
        print(denormalized_3[0][0][0])   
        
        # above values are basically the same values give or take small errors
        break

    
def evaluate():
    options = {
        "model_path": "checkpoints/save/",
        "ratios": [0.80, 0.10, 0.10],
        "date_range": (None, None),
        "market": "NASDAQ",
        "wiki_date": "20180105",
        "window_size": 30,
        "device": "cuda:1",
        "alpha": 0.6
    }
    options = edict(options)

        
    # 1. Load datasets
    """
    Here I'm loading the dataset in the exact same way as the training code.
    Mainly, notice that I've kept the "ratios" and the "date_range" the same.
    
    This gaurantees that we get the same valid_loader and test_loader as we did in 
    training time, which makes sure none of the training data is mixed in with the 
    valid loader or the test loader. 
    
    This is mainly possible because, split_dataset() is not a random process, by design. For a standard, usual
    random train_test_split, there's a function for that but I recommend against it: https://www.notion.so/dataset-concern-52801759c37d4c8fb8b7a82119c3a249

    When given "ratios", split_dataset(ratios) will split the data chronologically, not randomly:

    2021-06-07 10:05:12.883 | INFO     | tigre.datasets.stocks.StockDataset:split_dataset:56 - datasplit 0 start=2013-01-02 00:00:00 end=2016-12-29 00:00:00 count=1007
    2021-06-07 10:05:12.884 | INFO     | tigre.datasets.stocks.StockDataset:split_dataset:56 - datasplit 1 start=2016-12-30 00:00:00 end=2017-06-29 00:00:00 count=125
    2021-06-07 10:05:12.885 | INFO     | tigre.datasets.stocks.StockDataset:split_dataset:56 - datasplit 2 start=2017-06-30 00:00:00 end=2017-12-27 00:00:00 count=125
    """

    nasdaq = NASDAQStockDataset(date_range=options.date_range, normalize=True)
    wikigraph = WikiGraphDataset(
        market=options.market, 
        filter_tickers=nasdaq.get_tickers(), 
        date=options.wiki_date
    )

    # 2. Process data & build readers that model.fit() can utilize
    train, valid, test = nasdaq.split_dataset(ratios=options.ratios)
    relational_encoding, binary_encoding = wikigraph.get_encodings()
    relational_encoding = torch.Tensor(relational_encoding).to(options.device)

    valid_reader = StockWindowDataReader(valid, target_index=0, window_size=options.window_size)
    test_reader = StockWindowDataReader(test, target_index=0, window_size=options.window_size)

    valid_loader = valid_reader.get_dataloader(batch_size=2)
    test_loader = test_reader.get_dataloader(batch_size=2)

    # 3. Load model:
    model = TIGRE(
        model_path=options.model_path,
        device=options.device
    )
    evaluation = RankingMSELoss(alpha=options.alpha)
    
    for batch in test_loader:
        inputs = batch["inputs"].to(options.device)
        targets = batch["targets"].to(options.device)

        pred = model(inputs)
        print(pred.size())
        
        out = model(inputs, show_full_outputs=True)
        assert type(out) == list
        seq_embed, rel_embed, pred = out

        print(seq_embed.size())
        print(rel_embed.size())
        print(pred.size())
        loss = evaluation(pred, targets)
        print(loss)

        # denormalizing to see actual real-life results:
        denormalized_inputs = nasdaq.denormalize(inputs)
        


        break
    



def train():
    hyperparams = {
        "seq_embed_size": 16,
        "k_hops": 2,
        "hop_layers": 2,
        "lstm_layers": 2,
        "lstm_hidden_dim": 64,
        "fn_layers": 2,
        "h_size": 128,
        "double": False,
        "lr": 1e-4,
        "weight_decay": 1e-8,
        "alpha": 0.6,
        "window_size": 10,
        "epochs": 100,
        "batch_size": 8,
        "gradient_accumulation": 8
    }
    training_setup = {
        "ratios": [0.80, 0.10, 0.10],
        "date_range": (None, None),
        "market": "NASDAQ",
        "wiki_date": "20180105",
        "device": "cuda:1",
        "output_path": "checkpoints/two_save/",
        "evaluation_steps": 100
    }
    
    def forward(target):
        out = []
        for j in target:
            row = []
            for x in j:
                if x < 0:
                    row.append(-torch.pow(-x, 0.2))
                else:
                    row.append(torch.pow(x, 0.2))
            out.append(row)
        scaled = torch.tensor(out).to(options.device)
        return scaled + 0.5

    def backward(target):
        out = []
        for j in target:
            row = []
            for x in j:
                if x < 0:
                    row.append(-torch.pow(-x, 1 / 0.2))
                else:
                    row.append(torch.pow(x, 1 / 0.2))
            out.append(row)
        scaled = torch.tensor(out).to(options.device)
        return scaled - 0.5

    target_transform={
        "forward": forward,
        "backward": backward
    }
    options = {**hyperparams, **training_setup}
    options = edict(options)

    # 1. Load datasets
    nasdaq = NASDAQStockDataset(
        date_range=options.date_range,
        features=["price"], # price is a required feature.
        normalize=False
    )
    wikigraph = WikiGraphDataset(
        market=options.market, 
        filter_tickers=nasdaq.get_tickers(), 
        date=options.wiki_date
    )

    # 2. Process data & build readers that model.fit() can utilize
    train, valid, test = nasdaq.split_dataset(ratios=options.ratios)
    relational_encoding, binary_encoding = wikigraph.get_encodings()
    relational_encoding = torch.Tensor(relational_encoding).to(options.device)

    train_reader = StockWindowDataReader(stock_dataset=train, target_index=0, window_size=options.window_size, double=options.double)
    valid_reader = StockWindowDataReader(stock_dataset=valid, target_index=0, window_size=options.window_size, double=options.double)
    
    # 3. Define models
    input_data_shape, target_shape = train_reader.shape() # input_data_shape == (window_size, # of stocks, # of features per stock)

    sequential_embedding_model = LSTMSequentialEmbedding(input_data_shape, options.seq_embed_size, options.lstm_layers, options.lstm_hidden_dim)
    relational_embedding_model = FCRelationalEmbedding(sequential_embedding_model.output_shape(), relational_encoding, options.k_hops, options.hop_layers)
    prediction_model = FCPrediction(
        sequential_embedding_model.output_shape(), 
        relational_embedding_model.output_shape(), 
        n_layers=options.fn_layers,
        h_size=options.h_size,
        double=options.double
    )
    
    model = TIGRE(
        modules=(sequential_embedding_model, relational_embedding_model, prediction_model), 
        device=options.device,
        target_transform=target_transform
    )
    print(model)

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
    # evaluate()
    train()
