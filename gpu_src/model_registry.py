import cudf
import cupy as cp
from cuml.linear_model import ElasticNet
from sklearn.linear_model import Lars, LassoLars
from cuml.ensemble import RandomForestRegressor
from scipy.sparse import diags
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from cuml.svm import SVR
from cuml.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from cuml.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, RegressorMixin, clone
#from pytorch_tabular import TabularModel
#from pytorch_tabular.models import TabTransformerConfig
#from pytorch_tabular.config import (
    #DataConfig,
    #OptimizerConfig,
    #TrainerConfig,
#)

MODEL_REGISTRY = {
    "elastic_net": ElasticNet,
    "xg_boost": xgb.XGBRegressor,
    "random_forest": RandomForestRegressor,
    "support_vector": SVR
}

def create_model(name, params=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    if params == None:
        params = {}
    return MODEL_REGISTRY[name](**params)

class ClassifierGuidedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classifier, regressor_dict):
        self.classifier = classifier
        self.regressor_dict = regressor_dict # e.g., {0: LinearRegression(), 1: XGBRegressor()}
        self.regressors_ = {}

    def fit(self, X, y, sample_weight=None):
        y_labels = self._get_labels(y) 
        self.classifier_ = clone(self.classifier).fit(X, y_labels)
        
        # 2. Fit specialized regressors for each predicted class
        predicted_labels = self.classifier_.predict(X)
        for label, reg in self.regressor_dict.items():
            mask = (predicted_labels == label)
            if cp.any(mask):
                self.regressors_[label] = clone(reg).fit(X[mask], y[mask])
        return self

    def predict(self, X):
        # predict class label
        labels = self.classifier_.predict(X)
        predictions = cp.zeros(X.shape[0])
        
        # apply regressor
        for label, reg in self.regressors_.items():
            mask = (labels == label)
            if cp.any(mask):
                predictions[mask] = reg.predict(X[mask])
        return predictions

    def _get_labels(self, y):
        return (y > 0).astype(int)

MODEL_REGISTRY["hierarchical"] = ClassifierGuidedRegressor

class CustomFeatureAgg(FeatureAgglomeration):
    def __init__(self, *, n_clusters=2, metric="euclidean", connectivity=None, linkage="ward"):
        
        super().__init__(
            n_clusters=n_clusters, 
            metric=metric, 
            connectivity=connectivity,
            linkage=linkage
        )

    def fit(self, X, y=None): 
        if self.connectivity is None:
            N = X.shape[1]
            diag_1 = np.ones(N)
            diag_2 = np.ones(N-1)
            conn = diags([diag_2, diag_1, diag_2], offsets=[-1,0,1])
            self.connectivity = conn   
        return super().fit(X, y)

MODEL_REGISTRY["feature_agg"] = CustomFeatureAgg