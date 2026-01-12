import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lars, LassoLars
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
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
    "support_vector": SVR,
    "lasso_lars": LassoLars,
    "balanced_rf": BalancedRandomForestClassifier,
    "balanced_ada_boost": RUSBoostClassifier,
    "svc": SVC
}

def create_model(name, params=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    if params == None:
        params = {}
    if name == "hierarchical":
        classifier_name = params["classifier_name"]
        regressor_name = params["regressor_name"]
        classifier_params = params["classifier_params"]
        regressor_params = params["regressor_params"]
        classifier = create_model(classifier_name, classifier_params)
        regressor = create_model(regressor_name, regressor_params)
        return MODEL_REGISTRY[name](classifier=classifier, regressor=regressor)
    return MODEL_REGISTRY[name](**params)

class ClassifierGuidedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None):
        given_labels = self._get_labels(y)
        self.classifier_ = clone(self.classifier)
        self.regressor_ = clone(self.regressor)
        self.classifier_.fit(X,given_labels)
        mask_1 = given_labels == 1
        if not np.any(mask_1):
            self.regressor_ = DummyRegressor(strategy="constant", constant=0.0)
            self.regressor_.fit(X, y) 
            return self
        X_1, y_1 = X[mask_1,:], y[mask_1]
        self.regressor_.fit(X_1, y_1)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        self.labels = self.classifier_.predict(X)
        mask_1 = self.labels == 1
        X_1 = X[mask_1,:]
        y_pred = np.zeros(X.shape[0])
        y_pred[mask_1] = self.regressor_.predict(X_1)
        return y_pred

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