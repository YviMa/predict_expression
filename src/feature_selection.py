from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.decomposition import PCA


FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold,
    "scikitlearn_estimator": SelectFromModel,
    "pca": PCA
}

def get_feature_selector(name, estimator=None, params=None):
    if name not in FEATURE_SELECTION_REGISTRY:
        raise ValueError(f"Unknown feature selector: {name}") 
    if params is None:
        params = {}
    if name == "pca":
        return FEATURE_SELECTION_REGISTRY[name](**params)  
    if name == "sklearn_model":
        if estimator is None:
            raise ValueError("Estimator must be provided for sklearn_model feature selection")
        return FEATURE_SELECTION_REGISTRY[name](estimator, **params)  
    return FEATURE_SELECTION_REGISTRY[name](**params)
