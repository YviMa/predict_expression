from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
import tuning
from model_registry import create_model

FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold
}

class SelectWithSklearn(SelectFromModel):
    def __init__(self, X, y, config, *, threshold = 1e-5, prefit=False, norm_order = 1, max_features = None, importance_getter = "auto"):
        self.config = config
        self.prefit=prefit
        tune = self.config["preprocessing"]["feature_selection"]["tuning"]["tune"]
        estimator_name = self.config["preprocessing"]["feature_selection"]["estimator"]
        
        if tune == True:
            self.estimator = create_model(estimator_name)
            self.tune(X,y)
        else:
            params = self.config["preprocessing"]["feature_selection"]["params"]
            self.estimator = create_model(estimator_name, **params)

        super().__init__(self.estimator, threshold=threshold, prefit=self.prefit, norm_order=norm_order, max_features=max_features, importance_getter=importance_getter)


    def tune(self, X, y):
        tuning_config = self.config["preprocessing"]["feature_selection"]["tuning"]
        tuner_name = tuning_config["tuner"]
        tuner = tuning.get_tuner(tuner_name, {"estimator": self.estimator, "config": tuning_config})
        best_estimator, best_params = tuner.tune(X,y)
        self.estimator = best_estimator
        self.prefit = True

FEATURE_SELECTION_REGISTRY["sklearn_model"] = SelectWithSklearn