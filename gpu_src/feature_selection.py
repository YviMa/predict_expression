from cuml.feature_selection import SelectFromModel, VarianceThreshold
from cuml.base  import BaseEstimator, TransformerMixin


FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold,
    "scikitlearn_estimator": SelectFromModel
}

def get_feature_selector(name, estimator, params=None):
    if name not in FEATURE_SELECTION_REGISTRY:
        raise ValueError(f"Unknown feature selector: {name}")
    if params==None:
        params = {}
    return FEATURE_SELECTION_REGISTRY[name](estimator, **params)

class SelectFromTansformation(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
    def fit(self, X, y=None):
        return self.estimator.fit(X)
    def fit_transform(self, X, y = None):
        return self.estimator.fit_transform(X, y)
    def transform(self, X, y=None):
        return self.estimator.transform(X)
    def inverse_transform(self, X, y=None):
        return self.estimator.inverse_transform(X)

FEATURE_SELECTION_REGISTRY["scikitlearn_transformation"]=SelectFromTansformation