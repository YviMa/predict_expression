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
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None):
        given_labels = self._get_labels(y)
        self.classifier.fit(X,given_labels)
        mask_1 = given_labels == 1
        X_1, y_1 = X[mask_1,:], y[mask_1]
        self.regressor.fit(X_1, y_1)
        return self

    def predict(self, X):
        self.labels = self.classifier.predict(X)
        mask_1 = self.labels == 1
        X_1 = X[mask_1,:]
        y_pred = cp.zeros(X.shape[0])
        y_pred[mask_1] = self.regressor.predict(X_1)
        return y_pred

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