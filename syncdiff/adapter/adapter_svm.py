import os
import pandas as pd


class SVMSet(object):
    def __init__(self, data_folder: str, symptom: str):
        assert symptom in ["biaoli", "hanre", "xushi"]
        super(SVMSet, self).__init__()
        self._data_folder = data_folder
        self._symptom = symptom
        self._X, self._Y = None, None
        self._load_data()

    def _load_data(self):
        dataset = pd.read_csv(os.path.join(self._data_folder, "raw.csv"))
        self._X = dataset.iloc[:, 0:-3].values
        if self._symptom == "biaoli": 
            self._Y = dataset.iloc[:, -3].values
        elif self._symptom == "hanre": 
            self._Y = dataset.iloc[:, -2].values
        elif self._symptom == "xushi": 
            self._Y = dataset.iloc[:, -1].values
    
    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y
