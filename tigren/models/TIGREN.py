import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from loguru import logger
from typing import List
import tqdm
from tqdm import tqdm
from tqdm.autonotebook import trange

from tigren.datareaders import DataReader
from . import Prediction, RelationalEmbedding, SequentialEmbedding

class TIGREN_Wrapper(nn.Module):
    def __init__(
        self, 
        sequential_embedding_model: SequentialEmbedding, 
        relational_embedding_model: RelationalEmbedding, 
        prediction_model: Prediction, 
        device: str
    ):

        """Define the TIGREN (Temporally Inductive Graph Relational Embedding Network) model.
        Input to TIGREN is (#window size, # stocks, # features per stock). 
        Process:
            input => sequential_embedding_model => seq_embedding
            input, seq_embedding => relational_embedding_model => rel_embedding
            seq_embedding, rel_embedding => prediction_model => final_prediction_scores
            
        Args:
            sequential_embedding_model (SequentialEmbedding)
            relational_embedding_model (RelationalEmbedding)
            prediction_model (Prediction)
            device (str): cuda or cpu
        """
        super(TIGREN_Wrapper, self).__init__()
        self.device = device
        self.sequential_embedding_model = sequential_embedding_model.to(device)
        self.relational_embedding_model = relational_embedding_model.to(device)
        self.prediction_model = prediction_model.to(device)
        
    def forward(self, input_data):
        input_data.to(self.device)
        seq_embeddings = self.sequential_embedding_model(input_data)
        relational_embeddings = self.relational_embedding_model(seq_embeddings)
       
        predictions = self.prediction_model(seq_embeddings=seq_embeddings, relational_embeddings=relational_embeddings)
        predictions = predictions.squeeze(dim=-1)

        return predictions


class TIGREN(nn.Module):
    def __init__(
        self, 
        sequential_embedding_model: SequentialEmbedding, 
        relational_embedding_model: RelationalEmbedding, 
        prediction_model: Prediction, 
        device: str
    ):
        """Define the TIGREN (Temporally Inductive Graph Relational Embedding Network) model.
        Input to TIGREN is (#window size, # stocks, # features per stock). 
        Process:
            input => sequential_embedding_model => seq_embedding
            input, seq_embedding => relational_embedding_model => rel_embedding
            seq_embedding, rel_embedding => prediction_model => final_prediction_scores
            
        Args:
            sequential_embedding_model (SequentialEmbedding)
            relational_embedding_model (RelationalEmbedding)
            prediction_model (Prediction)
            device (str): cuda or cpu
        """
        super(TIGREN, self).__init__()
        self.device = device
        self.model = TIGREN_Wrapper(
            sequential_embedding_model,
            relational_embedding_model,
            prediction_model,
            device
        )

    def fit(
        self, 
        train_reader: DataReader, 
        valid_reader: DataReader,
        train_metric: torch.nn.Module,
        evaluation_metrics: List[torch.nn.Module],
        batch_size: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        output_path: str,
        evaluation_steps: int,
        save_best_model: bool,
        show_progress_bar: bool,
        gradient_accumulation: int = 1,
    ):
        '''Fit TIGREN model.

        Args:
            oh my god
        '''
        logger.add(os.path.join(output_path, "log.out"))

        if evaluation_steps % gradient_accumulation != 0:
            logger.warning("Evaluation steps is not a multiple of gradient_accumulation. This may lead to perserve interpretation of evaluations.")
        # 1. Set up training
        train_loader = train_reader.get_dataloader(batch_size=batch_size)
        valid_loader = valid_reader.get_dataloader(batch_size=batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_loss = 999999
        if output_path and not os.path.isdir(output_path):
            os.mkdir(output_path)

        training_steps = 0
        self.model.train()
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):

            epoch_train_loss = []
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar)):
                training_steps += 1
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)

                predicted_scores = self.model(inputs)
                loss = train_metric(predicted_scores, targets) / gradient_accumulation
                loss.backward()
                epoch_train_loss.append(loss.item() * gradient_accumulation) 
            
                if (training_steps - 1) % gradient_accumulation == 0 or training_steps == len(train_loader):
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()
                

                if training_steps % evaluation_steps == 0:
                    logger.info(f"Evaluation Step: {training_steps} (Epoch {epoch}) epoch train loss avg={np.mean(epoch_train_loss)}")
                    self._eval_during_training(
                        valid_loader=valid_loader,
                        evaluation_metrics=evaluation_metrics,
                        output_path=output_path,
                        save_best_model=save_best_model,
                        current_step=training_steps
                    )
       
            logger.info(f"Epoch {epoch} train loss avg={np.mean(epoch_train_loss)}")
            epoch_train_loss = []
            # One epoch done.
            self._eval_during_training(
                valid_loader=valid_loader,
                evaluation_metrics=evaluation_metrics,
                output_path=output_path,
                save_best_model=save_best_model,
                current_step=training_steps
            )

    def _eval_during_training(
        self, 
        valid_loader: torch.utils.data.DataLoader,
        evaluation_metrics: List[torch.nn.Module],
        output_path: str, 
        save_best_model: bool,
        current_step: int
    ):
        with torch.no_grad():
            losses = dict()
            for eval_metric in evaluation_metrics:
                eval_metric_name = eval_metric.__class__.__name__
                losses[eval_metric_name] = []
            
            for batch_idx, batch in enumerate(valid_loader):
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                predicted_scores = self.model(inputs)

                for eval_metric in evaluation_metrics:
                    eval_metric_name = eval_metric.__class__.__name__
                    loss = eval_metric(predicted_scores, targets)
                    losses[eval_metric_name].append(loss.item())

        sum_loss = 0
        for losskey in losses.keys():
            mean = np.mean(losses[losskey])
            losses[losskey] = mean
            sum_loss += mean
        losses["mean"] = sum_loss / len(losses.keys())
            
        df = pd.DataFrame([losses])
        df.index = [current_step]
        csv_path = os.path.join(output_path, "evaluation_results.csv" )
        df.to_csv(csv_path, mode="a", header=(not os.path.isfile(csv_path)), columns=losses.keys()) # append to current fike.
        
        if save_best_model:
            if sum_loss < self.best_loss:
                self.save(output_path)
                self.best_loss = sum_loss

    
    def save(self, path):
        if path is None:
            return

        os.makedirs(path, exist_ok=True)
        logger.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules["model"]._modules):
            module = self._modules["model"]._modules[name]
            model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            torch.save(module, os.path.join(model_path, "pytorch_model.bin"))
            
            module_parameters = module.config()
            if "large_data" in module_parameters.keys():
                for one_data_name in module_parameters["large_data"]:
                    one_large_data = module_parameters[one_data_name]
                    if not os.path.isfile(os.path.join(model_path, one_data_name+".pkl")):
                        logger.info(f"Pickling large data found in {model_path} config: {one_data_name}...")
                        pickle.dump(one_large_data, open(os.path.join(model_path, one_data_name + ".pkl"), "wb"))
                    del module_parameters[one_data_name]

            json.dump(module_parameters, open(os.path.join(model_path, "config.json"), "w"))
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

    def load(self, path: str):
        """Given the path to the model save directory, this will load TIGREN.
        """
        pass
        

