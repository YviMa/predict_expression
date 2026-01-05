import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
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
    "random_forest": RandomForestRegressor
}

def create_model(name, params=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    if params == None:
        params = {}
    return MODEL_REGISTRY[name](**params)


class TabularWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper around tabular model to fit scikit-learns distinction between X and y and 
    so it fits into scikit-learn pipelines.
    Only exists for the purpose of having one single tuning/evaluation file without
    case distinctions for data format based on the model.
    """
    def __init__(self, model_params, trainer_params, optimizer_params):

        self.model_params = model_params
        self.trainer_params = trainer_params
        self.optimizer_params = optimizer_params
        self.estimator = None
    

    def fit(self, X, y):
        y = pd.DataFrame(y)
        df = pd.DataFrame(X.copy())
        df[y.columns[0]] = y

        data_config = DataConfig(
            target=[
                y.columns[0]
            ],  # target should always be a list
            continuous_cols=list(X.columns),
            categorical_cols=[],
            normalize_continuous_features = False,
        )

        model_config = TabTransformerConfig(**self.model_params)
        optimizer_config = OptimizerConfig(**self.optimizer_params)
        trainer_config = TrainerConfig(**self.trainer_params)

        self.estimator = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
        )
        self.estimator.fit(train=df)
        return self

    def predict(self, X):
        preds = self.estimator.predict(X)
        return preds

MODEL_REGISTRY["tabtransformer"] = TabularWrapper
