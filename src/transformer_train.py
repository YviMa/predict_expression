from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import TabularModel
from pytorch_tabular import TabularModelTuner
from torchmetrics import MeanSquaredError
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from os.path import join
from utils import load_config, set_up_experiment, load_data, compute_metrics, plot_training_results, custom_metric
from scaling import apply_scaling
from feature_selection import get_feature_selector
from sklearn.base import BaseEstimator
from itertools import product
import torch


print(torch.cuda.is_available())

#parse the yaml file 
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# load yaml config file with all the options you provided
config = load_config(args.config)

# creates separate directory for each experiment
exp_dir = set_up_experiment(config)

data= pd.read_csv(os.path.join(config['data']['data_dir'], config['data']['file_name']), sep='\t')

data.iloc[:,-1]=np.log2(data.iloc[:,-1]+1)
target_col = "Expression"
num_col_names = list(data.columns[1:-1])
cat_col_names = []

data_config = DataConfig(
    target=[
        target_col
    ],  # target should always be a list
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    batch_size=128,
    max_epochs=300,
)
optimizer_config = OptimizerConfig()

model_config = TabTransformerConfig(
    task="regression",
    learning_rate=1e-3,
)

'''
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True
)
'''

param_grid = config["training"]["param_grid"]
cv = config["training"].get("cv_folds", 5)
scoring = config["training"].get("scoring", None)


tuner = TabularModelTuner(
   data_config=data_config,
   model_config=model_config,
   optimizer_config=optimizer_config,
   trainer_config=trainer_config,
   verbose=True
)

trials_df, best_params, best_score, best_model =tuner.tune(
    train = data, 
    search_space = param_grid, 
    metric = "mean_squared_error",
    mode = "min",
    strategy ="grid_search",
    cv = 5,
    verbose = True
)



with open(os.path.join(exp_dir,"tuned_params.json"), "w") as f:
    json.dump({"best_params":best_params, "best_score": best_score}, f, indent=2)