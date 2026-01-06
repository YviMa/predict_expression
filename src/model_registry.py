import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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


class TabularWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper around tabular model to fit scikit-learns distinction between X and y and 
    so it fits into scikit-learn pipelines.
    Only exists for the purpose of having one single tuning/evaluation file without
    case distinctions for data format based on the model.
    """
    def __init__(self, optimizer='Adam', learning_rate=1e-3, num_attn_blocks=2, ff_hidden_multiplier=4, transformer_activation='LeakyReLU', embedding_dropout=0.1, task="regression", batch_size=32, max_epochs=100):

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_attn_blocks = num_attn_blocks
        self.ff_hidden_multiplier = ff_hidden_multiplier
        self.transformer_activation = transformer_activation
        self.embedding_dropout = embedding_dropout
        self.task = task
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.estimator = None
    
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            # Use stored feature names if they exist, else generic ones
            cols = self.feature_names if hasattr(self, "feature_names") else [f"feat_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
    
        # 2. FORCE y TO SERIES AND ALIGN INDEX
        # Even if X was NumPy, we just turned it into a DF, so it has an index now
        if not isinstance(y, pd.Series):
            # We flatten() here only to make the pd.Series happy
            y_flat = np.asanyarray(y).flatten()
            y = pd.Series(y_flat, index=X.index, name="Expression")
        df = X.copy()
        df["Expression"] = y

        data_config = DataConfig(
            target=[
                y.columns[0]
            ],  # target should always be a list
            continuous_cols=list(X.columns),
            categorical_cols=[],
        )

        model_config = TabTransformerConfig(
            task = self.task,
            learning_rate = self.learning_rate,
            num_attn_blocks = self.num_attn_blocks,
            ff_hidden_multiplier = self.ff_hidden_multiplier,
            transformer_activation = self.transformer_activation,
            embedding_dropout = self.embedding_dropout,
        )
        optimizer_config = OptimizerConfig(
            optimizer = self.optimizer,
        )
        trainer_config = TrainerConfig(
            batch_size = self.batch_size,
            max_epochs = self.max_epochs,
            trainer_kwargs = {"enable_progress_bar": False} # very important if working on remote, do not change, training will crash
        )                                                   # due to missing display

        self.estimator_ = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
        )
        self.estimator_.fit(train=df)
        return self

    def predict(self, X):
        preds = self.estimator_.predict(X)
        return preds

MODEL_REGISTRY["tabtransformer"] = TabularWrapper

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
