import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lars, LassoLars
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.base import BaseEstimator, RegressorMixin, clone
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

MODEL_REGISTRY = {
    "elastic_net": ElasticNet,
    "gradient_boost": GradientBoostingRegressor,
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
            if np.any(mask):
                self.regressors_[label] = clone(reg).fit(X[mask], y[mask])
        return self

    def predict(self, X):
        # predict class label
        labels = self.classifier_.predict(X)
        predictions = np.zeros(X.shape[0])
        
        # apply regressor
        for label, reg in self.regressors_.items():
            mask = (labels == label)
            if np.any(mask):
                predictions[mask] = reg.predict(X[mask])
        return predictions

    def _get_labels(self, y):
        return (y > 0).astype(int)

MODEL_REGISTRY["hierarchical"] = ClassifierGuidedRegressor

class CustomFeatureAgg(FeatureAgglomeration):
    def __init__(self, n_clusters = 2, *, metric = "euclidean", memory = None, connectivity = None, compute_full_tree = "auto", linkage = "ward", pooling_func = np.mean, distance_threshold = None, compute_distances = False):
        super().__init__(n_clusters, metric=metric, memory=memory, connectivity=connectivity, compute_full_tree=compute_full_tree, linkage=linkage, pooling_func=pooling_func, distance_threshold=distance_threshold, compute_distances=compute_distances)

    def fit(self, X, y=None):
        N = X.shape[1]
        conn = np.tri(N)
        conn[np.diag_indices(N)]=0
        self.connectivity=conn
        return super().fit(X,y)

MODEL_REGISTRY["feature_agg"] = CustomFeatureAgg