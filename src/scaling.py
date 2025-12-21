import numpy as np
from sklearn.preprocessing import StandardScaler

SCALER_REGISTRY = {
    "standard": StandardScaler,
}

def get_scaler(name, params={}):
    cls = SCALER_REGISTRY[name]
    return cls(**params)

class LogScaler:
    def fit(self, X):
        return self
    def transform(self, X):
        return np.log2(X+1)
    def fit_transform(self, X):
        return self.fit(X).transform(X)

SCALER_REGISTRY["log2"] = LogScaler