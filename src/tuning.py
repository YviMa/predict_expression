import pandas as pd
from sklearn.model_selection import GridSearchCV
from pytorch_tabular import TabularModelTuner
from model_registry import create_model
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

TUNING_REGISTRY = {}

def get_tuner(name, estimator, params=None):
    if name not in TUNING_REGISTRY:
        raise ValueError(f"Unknown tuner: {name}")
    if params is None:
        params = {}
    params.update({"estimator": estimator})
    return TUNING_REGISTRY[name](**params)


class TabularModelTunerWrapper:

    def __init__(self, estimator, model, trainer, search_space, metric, mode, strategy, cv, verbose=True):

        self.estimator_name = estimator_name
        self.tuner = None
        self.search_space = search_space
        self.metric = metric
        self.mode = mode
        self.strategy = strategy
        self.cv = cv
        self.verbose = verbose


        #target_range = [(-1 , 15)]
        

        self.trainer_config = TrainerConfig(
            batch_size=trainer["batch_size"],
            max_epochs=trainer["max_epochs"],
        )
        self.optimizer_config = OptimizerConfig()

        self.model_config = TabTransformerConfig(
            task=model["task"],
            #target_range=target_range,
        )

        self.data_config = None

    def tune(self, X, y):
        y = pd.DataFrame(y)
        df = pd.DataFrame(X.copy())
        df[y.columns[0]] = y
        self.data_config = DataConfig(
            target=[
                y.columns[0]
            ],  # target should always be a list
            continuous_cols=list(X.columns),
            categorical_cols=[],
            normalize_continuous_features=False,
        )

        self.tuner = TabularModelTuner(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
            verbose=True
            )

        trials_df, best_params, best_score, best_model = self.tuner.tune(
            train = df, 
            search_space = self.search_space, 
            metric = self.metric,
            mode = self.mode,
            strategy = self.strategy,
            cv = self.cv,
            verbose = True
        )

        return trials_df, best_params, best_score, best_model



TUNING_REGISTRY["tab_tuner"]= TabularModelTunerWrapper