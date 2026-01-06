import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lars, LassoLars
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
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

class SelectWithPCA(BaseEstimator):
    def __init__(self, n_components = None):
        super().__init__()
        self.n_components = n_components
    def fit(self, X, y = None):
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X)
        self.feature_importances_ = np.abs(self.pca_.components_).sum(axis=0)
        return self

MODEL_REGISTRY["pca_selector"] = SelectWithPCA # to use with scikitlearn's SelectFromModel, NOT to be used independently