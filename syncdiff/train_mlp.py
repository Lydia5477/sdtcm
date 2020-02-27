import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from .models import MLP
from .adapter import MLPSet
from datetime import datetime


class MLPTrainer(object):
    def __init__(self, opt: dict):
        super(MLPTrainer, self).__init__()
        self._opt = opt
        self._model = MLP(opt["input_size"], opt["output_size"])
        self._optimizer = optim.Adam(self._model.parameters(), opt["learning_rate"])
        self._loss_fn = nn.CrossEntropyLoss()
        self._train_set, self._train_loader = None, None
        self._test_set, self._test_loader = None, None
    
    def _eval(self):
        self._model.eval()
        y_true, y_pred = [], []
        for _, (x, y) in enumerate(self._test_loader):
            y_ = self._model(x)
            y_pred.extend(y_.argmax(dim=-1).tolist())
            y_true.extend(y.tolist())

        return y_true, y_pred
    
    def _load_loaders(self, what_fold: int, symptom: str):
        self._train_set = MLPSet(
            self._opt["data_folder"], "train", symptom, self._opt["cross_validation_fold"], what_fold)
        self._test_set = MLPSet(
            self._opt["data_folder"], "test", symptom, self._opt["cross_validation_fold"], what_fold)
        self._train_loader = DataLoader(
            dataset=self._train_set,
            batch_size=self._opt["batch_size"],
            num_workers=0
        )
        self._test_loader = DataLoader(
            dataset=self._test_set,
            batch_size=self._opt["batch_size"],
            num_workers=0
        )
    
    def train_mlp(self, symptom: str):
        total_accuracy = 0.
        for what_fold in range(self._opt["cross_validation_fold"]):
            self._load_loaders(what_fold, symptom)
            max_accuracy, no_improvement = -1, 0
            for e in range(self._opt["epoch"]):
                total_loss = 0.
                self._model.train()
                for _, (x, y) in enumerate(self._train_loader):
                    self._model.zero_grad()
                    y_ = self._model(x)
                    loss = self._loss_fn(y_.view(-1, self._opt["output_size"]), y.view(-1))
                    loss.backward()
                    self._optimizer.step()
                    total_loss += loss.item()
                y_true, y_pred = self._eval()
                f1 = f1_score(y_true, y_pred, average="micro")
                precesion = precision_score(y_true, y_pred, average="micro")
                recall = recall_score(y_true, y_pred, average="micro")
                accuracy = accuracy_score(y_true, y_pred)
                print("[MLP] epoch {0}, f1: {1}, precesion: {2}, recall: {3}, accuracy: {4}, avg-loss: {5}".format(
                    e, round(f1, 5), round(precesion, 5), round(recall, 5), round(accuracy, 5), round(total_loss/len(self._train_loader), 5)
                ))
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement > self._opt["stop_if_no_improvemnet"]:
                    break
            total_accuracy += max_accuracy
            print("#[MLP] Fold: {0}, Train {1} with accuracy {2}".format(
                what_fold, symptom, round(max_accuracy, 5)))
        avg_accuracy = total_accuracy / self._opt["cross_validation_fold"]

        return avg_accuracy

    def train(self):
        result, total_accuracy = {}, 0.
        for symptom in self._opt["symptoms"]:
            start = datetime.now()
            avg_accuracy = self.train_mlp(symptom)
            print("\033[92m# Train model MLP on syndrome differentiation {0} with accuracy {1}\033[0m".format(
                symptom, round(avg_accuracy, 5)))
            total_accuracy += avg_accuracy
            result[symptom] = {
                "accuracy": avg_accuracy,
                "time": (datetime.now() - start).total_seconds(),
                "batch_size": self._opt["batch_size"], 
                "learning_rate": self._opt["learning_rate"]
            }
        result["avg_accuracy"] = total_accuracy / len(self._opt["symptoms"])
        
        return result
