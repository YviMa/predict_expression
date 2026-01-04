import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCALER_REGISTRY = {
    "standard": StandardScaler,
}

def get_scaler(name, params={}):
    cls = SCALER_REGISTRY[name]
    return cls(**params)

def apply_scaling(X_train, scaler_list):

    for item in scaler_list:
        scaler = get_scaler(item["scaler"], item["params"])
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test
    

class LogScaler:
    def fit(self, X):
        return self
    def transform(self, X):
        return np.log2(X+1)
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        return 2**(X)-1

SCALER_REGISTRY["log2"] = LogScaler

class ScalerWrapper():
    def __init__(self, scaler_names):
        self.scaler_list = []
        for name in scaler_names:
            self.scaler_list.append(get_scaler(name))
    
    def fit(self, X):
        X_ = X
        for scaler in self.scaler_list:
            scaler.fit(X_)
            X_ = scaler.transform(X_)
        return self

    def transform(self, X):
        columns = X.columns
        index = X.index
        X_ = X.copy()
        for scaler in self.scaler_list:
            X_ = scaler.transform(X_)
        df = pd.DataFrame(X_, index=index, columns=columns)
        return df
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        columns = X.columns
        index = X.index
        X_ = X.copy()
        for scaler in reversed(self.scaler_list):
            X_ = scaler.inverse_transform(X_)
        df = pd.DataFrame(X_, index=index, columns=columns)
        return df