from sklearn.feature_selection import SelectFromModel, VarianceThreshold


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