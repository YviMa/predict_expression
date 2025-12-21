from sklearn.model_selection import GridSearchCV

TUNING_REGISTRY = {}

def get_tuner(name, params=None):
    if name not in TUNING_REGISTRY:
        raise ValueError(f"Unknown tuner: {name}")
    return TUNING_REGISTRY[name](**params)

class BaseTuner:
    def __init__(self, estimator, config):
        self.estimator = estimator
        self.config = config

    def tune(self, X, y):
        """Return best estimator or best parameters"""
        raise NotImplementedError

class SklearnGridTuner(BaseTuner):
    def tune(self, X, y):
        param_grid = self.config["param_grid"]
        cv = self.config.get("cv_folds", 5)
        scoring = self.config.get("scoring", None)
        grid = GridSearchCV(
            estimator=self.estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            refit = True,
            n_jobs=-1
        )
        grid.fit(X, y)
        return grid.best_estimator_, grid.best_params_

TUNING_REGISTRY["gridsearch_cv"] = SklearnGridTuner