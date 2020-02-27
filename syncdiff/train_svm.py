from .models import SVM
from .adapter import SVMSet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class SVMTrainer(object):
    def __init__(self, opt: dict):
        super(SVMTrainer, self).__init__()
        self._opt = opt
        self._dataset = None
    
    def _load_dataset(self, symptom: str):
        self._dataset = SVMSet(self._opt["data_folder"], symptom)
    
    def train_svm(self, symptom: str):
        kf = KFold(n_splits=self._opt["cross_validation_fold"])
        total_accuracy = 0.
        for train_idx, test_idx in kf.split(self._dataset.X):
            X_train, X_test = self._dataset.X[train_idx], self._dataset.X[test_idx]
            y_train, y_test = self._dataset.Y[train_idx], self._dataset.Y[test_idx]
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            classifier = SVM(kernel = 'rbf', random_state = 0, gamma='auto')
            classifier.fit(X_train, y_train)
            test_accuracy = classifier.score(X_test, y_test)
            total_accuracy += test_accuracy
        avg_accuracy = total_accuracy / self._opt["cross_validation_fold"]

        return avg_accuracy
    
    def train(self):
        result, total_accuracy = {}, 0.
        for symptom in self._opt["symptoms"]:
            self._load_dataset(symptom)
            start = datetime.now()
            avg_accuracy = self.train_svm(symptom)
            print("\033[92m# Train model SVM on syndrome differentiation {0} with accuracy {1}\033[0m".format(
                symptom, round(avg_accuracy, 5)))
            total_accuracy += avg_accuracy
            result[symptom] = {
                "accuracy": avg_accuracy,
                "time": (datetime.now() - start).total_seconds()
            }
        result["avg_accuracy"] = total_accuracy / len(self._opt["symptoms"])
        
        return result
