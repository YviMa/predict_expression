from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
import tuning
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from model_registry import create_model

FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold
}

def get_feature_selector(name, params=None):
    if name not in FEATURE_SELECTION_REGISTRY:
        raise ValueError(f"Unknown feature selector: {name}")
    if params==None:
        params = {}
    return FEATURE_SELECTION_REGISTRY[name](**params)

class SelectWithSklearn(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_name, estimator_params):
        super().__init__()
        self.estimator_name = estimator_name
        self.estimator_params = estimator_params
        self.estimator = None
    def fit(self, X, y=None):
        self.estimator = create_model(self.estimator_name, self.estimator_params)
        self.estimator.fit(X, y)
        return self
    def transform(self, X):
        return self.estimator.transform(X)
    def fit_transform(self, X, y = None):
        return self.estimator.fit_transform(X)
    def get_feature_names_out(self, input_features=None):
        return self.estimator.get_feature_names_out(input_features)
    
'''
class SelectWithSklearn(SelectFromModel):
    def __init__(self, X, y, config, *, threshold = 1e-5, prefit=False, norm_order = 1, max_features = None, importance_getter = "auto"):
        self.config = config
        self.X = X
        self.y = y
        tune = self.config["tuning"]["tune"]
        estimator_name = self.config["estimator"]
        
        if tune == True:
            self.estimator = create_model(estimator_name)
            self.tune(X,y)
        else:
            params = self.config["params"]
            self.estimator = create_model(estimator_name, params)

        super().__init__(self.estimator, threshold=threshold, prefit=prefit, norm_order=norm_order, max_features=max_features, importance_getter=importance_getter)


    def tune(self, X, y):
        tuning_config = self.config["tuning"]
        tuner_name = tuning_config["tuner"]
        tuner = tuning.get_tuner(tuner_name, {"estimator": self.estimator, "config": tuning_config})
        best_estimator, best_params = tuner.tune(X,y)
        self.estimator = best_estimator
        self.prefit = True
'''
class SelectWithSklearn(SelectFromModel):
    def __init__(
        self,
        config,
        *,
        threshold=1e-5,
        prefit=False,
        norm_order=1,
        max_features=None,
        importance_getter="auto",
    ):
        self.config = config
        self.prefit = prefit

        tune = config["tuning"]["tune"]
        estimator_name = config["estimator"]

        if tune:
            self.estimator = create_model(estimator_name)
        else:
            self.estimator = create_model(estimator_name, config["params"])

        super().__init__(
            self.estimator,
            threshold=threshold,
            prefit=prefit,
            norm_order=norm_order,
            max_features=max_features,
            importance_getter=importance_getter,
        )

FEATURE_SELECTION_REGISTRY["sklearn_model"] = SelectWithSklearn