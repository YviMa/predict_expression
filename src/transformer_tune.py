import argparse
import matplotlib.pyplot as plt
import json
import ast
import itertools
import numpy as np
import pandas as pd
from scaling import build_scaler
from os.path import join
from sklearn.model_selection import KFold
from utils import load_config, set_up_experiment, load_data, log2p1, exp2m1, split_params
from model_registry import create_model
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import TabularModel


# parse the yaml file 
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# load yaml config file with all the options you provided
config = load_config(args.config)

# creates separate directory for each experiment
exp_dir = set_up_experiment(config)

data_path = join(config['data']['data_dir'], config['data']['file_name'])
data = pd.read_csv(data_path, sep='\t')
num_col_names = list(data.columns[1:-1])
cat_col_names = []
target_col = ["Expression"]

est_params = config["estimator"]['est_params']

fixed_model_params, fixed_trainer_params, fixed_optimizer_params = split_params(est_params)

outer_kf = KFold(n_splits=2, shuffle = True, random_state = 12)
inner_kf = KFold(n_splits=2, shuffle = True, random_state = 8)


param_grid = config["tuning"]["param_grid"]

results = []
for (outer_train_index, outer_test_index) in outer_kf.split(data):

    train_data = data.iloc[outer_train_index,:]
    test_data = data.iloc[outer_test_index,:]

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    combo_rmses = []
    for combo in param_combinations:
        rmses = []
        for (inner_train_index, inner_test_index) in inner_kf.split(train_data):

            tune_data  = train_data.iloc[inner_train_index, :]
            val_data = train_data.iloc[inner_train_index, :]

            # by default, pytorch_tabular's models perform standard scaling so only log scale here
            if config["preprocessing"]["y_scaling"] == "log2":
                logged = log2p1(tune_data["Expression"])
                tune_data["Expression"] = logged

            tune_model_params, tune_trainer_params, tune_optimizer_params = split_params(combo)

            model_params = {**fixed_model_params, **tune_model_params}
            trainer_params = {**fixed_trainer_params, **tune_trainer_params}
            optimizer_params = {**fixed_optimizer_params, **tune_optimizer_params}

            data_config = DataConfig(
                target=target_col,
                continuous_cols=num_col_names,
                categorical_cols=cat_col_names,
                num_workers=0
            )
            trainer_config = TrainerConfig(**trainer_params, 
                                           trainer_kwargs = {"enable_progress_bar": False, # very important, if disabled will produce error due to missing display
                                            "log_every_n_steps": 2}) 
                                                                                            
            optimizer_config = OptimizerConfig(**optimizer_params)
            model_config = TabTransformerConfig(**model_params)

            estimator = TabularModel(
                data_config=data_config,
                model_config=model_config,
                trainer_config=trainer_config,
                optimizer_config=optimizer_config,
            )

            estimator.fit(train = tune_data)
            
            pred_df = estimator.predict(val_data)

            # scale back to original scale for validation
            if config["preprocessing"]["y_scaling"] == "log2":
                expd = exp2m1(pred_df["Expression_prediction"])
                pred_df["Expression_prediction"] = expd
            
            y_pred_raw=pred_df["Expression_prediction"].values.ravel()
            y_test_raw=val_data["Expression"].values.ravel()

            rmse = root_mean_squared_error(y_test_raw, y_pred_raw)
            rmses.append(rmse)
        
        mean_rmse = np.array(rmses).mean()
        new_row = {**combo, **{"RMSE": mean_rmse}}
        combo_rmses.append(new_row)
        
    best_params_with_rmse = min(combo_rmses, key=lambda x: x['RMSE'])
    best_params = best_params_with_rmse.copy()
    del best_params['RMSE']

    best_model_params, best_trainer_params, best_optimizer_params = split_params(best_params)

    model_params = {**fixed_model_params, **best_model_params}
    trainer_params = {**fixed_trainer_params, **best_trainer_params}
    optimizer_params = {**fixed_optimizer_params, **best_optimizer_params}

    data_config = DataConfig(
        target=target_col,
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )

    trainer_config = TrainerConfig(**trainer_params, trainer_kwargs = {"enable_progress_bar": False})
    optimizer_config = OptimizerConfig(**optimizer_params)
    model_config = TabTransformerConfig(**model_params)

    best_estimator = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
    )

    if config["preprocessing"]["y_scaling"] == "log2":
        logged = log2p1(train_data["Expression"])
        train_data["Expression"] = logged

    best_estimator.fit(train =train_data)
    pred_df = best_estimator.predict(test_data)

    if config["preprocessing"]["y_scaling"] == "log2":
        expd = exp2m1(pred_df["Expression_prediction"])
        pred_df["Expression_prediction"] = expd
            
    y_pred_raw=pred_df["Expression_prediction"].values.ravel()
    y_test_raw=test_data["Expression"].values.ravel()

    rmse = root_mean_squared_error(y_test_raw, y_pred_raw)
    r_val, _ = pearsonr(y_test_raw, y_pred_raw)

    results.append({"best_combo": str(best_params), "RMSE": rmse, "pearsonr": r_val})

df = pd.DataFrame(results)
df.to_csv(join(exp_dir, "nested_crossval_results.csv"), sep='\t')

mean = df[["RMSE", "pearsonr"]].mean()
std = df[["RMSE", "pearsonr"]].std()

total_accuracy = pd.DataFrame({"mean": mean, "std": std})
total_accuracy.to_csv(join(exp_dir, "nested_cross_val_accuracy.csv"), sep='\t')

kf = KFold(n_splits=5, shuffle = True, random_state = 9)

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

combo_rmses = []
for combo in param_combinations:
    rmses = []
    for (train_index, test_index) in kf.split(data):

        tune_data  = data.iloc[train_index, :]
        val_data = data.iloc[train_index, :]

        # by default, pytorch_tabular's models perform standard scaling so only log scale here
        if config["preprocessing"]["y_scaling"] == "log2":
            logged = log2p1(tune_data["Expression"])
            tune_data["Expression"] = logged

        tune_model_params, tune_trainer_params, tune_optimizer_params = split_params(combo)

        model_params = {**fixed_model_params, **tune_model_params}
        trainer_params = {**fixed_trainer_params, **tune_trainer_params}
        optimizer_params = {**fixed_optimizer_params, **tune_optimizer_params}

        data_config = DataConfig(
            target=target_col,
            continuous_cols=num_col_names,
            categorical_cols=cat_col_names,
            num_workers=0
        )
        trainer_config = TrainerConfig(**trainer_params, 
                                        trainer_kwargs = {"enable_progress_bar": False, # very important, if disabled will produce error due to missing display
                                        "log_every_n_steps": 2}) 
                                                                                        
        optimizer_config = OptimizerConfig(**optimizer_params)
        model_config = TabTransformerConfig(**model_params)

        estimator = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
        )

        estimator.fit(train = tune_data)
        
        pred_df = estimator.predict(val_data)

        # scale back to original scale for validation
        if config["preprocessing"]["y_scaling"] == "log2":
            expd = exp2m1(pred_df["Expression_prediction"])
            pred_df["Expression_prediction"] = expd
        
        y_pred_raw=pred_df["Expression_prediction"].values.ravel()
        y_test_raw=val_data["Expression"].values.ravel()

        rmse = root_mean_squared_error(y_test_raw, y_pred_raw)
        rmses.append(rmse)
    mean_rmse = np.array(rmses).mean()
    new_row = {**combo, **{"RMSE": mean_rmse}}
    combo_rmses.append(new_row)

overall_best_params_with_rmse = min(combo_rmses, key=lambda x: x['RMSE'])
overall_best_params = overall_best_params_with_rmse.copy()
del overall_best_params['RMSE']

with open(join(exp_dir,"tuned_params.json"), "w") as f:
    json.dump(overall_best_params, f, indent=2)

