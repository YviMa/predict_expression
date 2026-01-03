from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import TabularModel


data_config = DataConfig(
    target=[
        target_col
    ],  # target should always be a list
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    batch_size=1024,
    max_epochs=100,
)
optimizer_config = OptimizerConfig()
model_config = TabTransformerConfig(
    task="classification",
    gflu_stages=6,
    gflu_feature_init_sparsity=0.3,
    gflu_dropout=0.0,
    learning_rate=1e-3,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True
)