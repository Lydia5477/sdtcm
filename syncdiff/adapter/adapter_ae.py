from torch.utils.data import Dataset
import os
import json
import torch


class AESet(Dataset):
    def __init__(self, data_folder: str):
        super(AESet, self).__init__()
        self._data_folder = data_folder
        self._input_size = 0
        self._data = []
        self._load_data()

    def _load_data(self):
        with open(os.path.join(self._data_folder, "data.json"), "r", encoding="utf-8") as f: 
            data = json.loads(f.read())
            data = [(data["inputs"][i], [t + 1 for t in data["inputs"][i]]) for i in range(len(data["inputs"]))]
            if len(data) > 0: self._input_size = len(data[0][0])
        self._data.extend(data)
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]

        return torch.tensor(item[0]).float(), torch.tensor(item[1])
