from torch.utils.data import Dataset
import os
import json
import torch


class MLPSet(Dataset):
    def __init__(self, data_folder: str, mode: str, symptom: str, cv_fold: int, what_fold: int):
        assert mode in ["train", "test"]
        assert symptom in ["biaoli", "hanre", "xushi"]
        super(MLPSet, self).__init__()
        self._data_folder = data_folder
        self._mode = mode
        self._symptom = symptom
        self._cv_fold = cv_fold
        self._what_fold = what_fold
        self._input_size = 0
        self._data = []
        self._load_data()

    def _load_data(self):
        with open(os.path.join(self._data_folder, "data.json"), "r", encoding="utf-8") as f: 
            data = json.loads(f.read())
            data = [(data["inputs"][i], data[self._symptom][i] + 1) for i in range(len(data["inputs"]))]
            if len(data) > 0: self._input_size = len(data[0][0])
        step_len = len(data) // self._cv_fold
        if self._mode == "test":
            if self._what_fold == self._cv_fold - 1:
                self._data.extend(data[self._what_fold*step_len: ])
            else:
                self._data.extend(data[self._what_fold*step_len: (self._what_fold+1)*step_len])
        else:
            for i in range(self._cv_fold):
                if i != self._what_fold:
                    if i == self._cv_fold - 1:
                        self._data.extend(data[i*step_len: ])
                    else:
                        self._data.extend(data[i*step_len: (i+1)*step_len])
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]

        return torch.tensor(item[0]).float(), torch.tensor(item[1])
