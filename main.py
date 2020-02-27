import fire
from syncdiff.adapter import data_partition
from syncdiff.utils import load_opt
from syncdiff.utils import combine_hps
from syncdiff.vist import visualize_raw_data
from syncdiff import AETrainer
from syncdiff import MLPTrainer
from syncdiff import SVMTrainer
from syncdiff.vist import dataset_stat
import torch
import numpy as np
import pprint


def _train_svm(opt: dict):
    trainer = SVMTrainer(opt)
    result = trainer.train()

    return result


def _train_ae(opt: dict):
    trainer = AETrainer(opt)
    result = trainer.train()

    return result


def _train_mlp(opt: dict):
    trainer = MLPTrainer(opt)
    result = trainer.train()

    return result


TRAINER_MAP = {
    "AE": AETrainer,
    "MLP": MLPTrainer
}


class CLI(object):
    def preprocess(self, data_folder: str = "./syncdiff/data/"):
        data_partition(data_folder)
    
    def dataset_stat(self, dataset_path: str = "./syncdiff/data/raw.csv"):
        result = dataset_stat(dataset_path)
        pprint.pprint(result)
    
    def finetuning(self, opt_path: str = "./config/default.json"):
        opt = load_opt(opt_path)
        results = {}
        for model_name, Trainer in TRAINER_MAP.items():
            if model_name == "AE": 
                hps = combine_hps(True)
            else:
                hps = combine_hps(False)
            for hp in hps:
                for k, v in hp.items():
                    opt[k] = v
                    trainer = Trainer(opt)
                    result = trainer.train()
                    if model_name not in results.keys():
                        results[model_name] = result
                    elif results[model_name]["avg_accuracy"] < result["avg_accuracy"]:
                        results[model_name] = result
        results["SVM"] = _train_svm(opt)
        pprint.pprint(results)
    
    def train(self, opt_path: str = "./config/default.json"):
        opt = load_opt(opt_path)
        results = {}
        results["SVM"] = _train_svm(opt)
        results["AE"] = _train_ae(opt)
        results["MLP"] = _train_mlp(opt)
        pprint.pprint(results)
    
    def train_svm(self, opt_path: str = "./config/default.json"):
        opt = load_opt(opt_path)
        result = _train_svm(opt)
        pprint.pprint(result)
    
    def train_ae(self, opt_path: str = "./config/default.json"):
        opt = load_opt(opt_path)
        result = _train_ae(opt)
        pprint.pprint(result)

    def train_mlp(self, opt_path: str = "./config/default.json"):
        opt = load_opt(opt_path)
        result = _train_mlp(opt)
        pprint.pprint(result)
        
    def visualize(self, opt_path: str = "./config/default.json" , data_path: str = "./syncdiff/data/data.json"):
        opt = load_opt(opt_path)
        visualize_raw_data(data_path, opt["saved_folder"])


if __name__ == "__main__":
    torch.manual_seed(9999)
    np.random.seed(9999)
    fire.Fire(CLI)
