import json
import joblib
import gc
import time
import argparse
from os.path import join
from sklearn.model_selection import KFold
import itertools
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
import cupy as cp
import numpy as np
import pandas as pd
import cudf
import cuml
from scaling import build_scaler
from sklearn.preprocessing import StandardScaler
from utils import load_config, set_up_experiment, load_data, log2p1, exp2m1, split_sk_params
from model_registry import create_model
#from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from cuml.model_selection import GridSearchCV
from scipy.stats import pearsonr
from feature_selection import get_feature_selector
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

''' 
----MEMORY STUFF------------------------------------------------------
'''
cuml.internals.logger.set_level(4)

rmm.reinitialize(
    pool_allocator=True, 
    initial_pool_size=None, 
    managed_memory=True
)

cp.cuda.set_allocator(rmm_cupy_allocator)

gc.collect()

cp.cuda.runtime.deviceSynchronize() # Force all kernels to finish
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
'''
----------------------------------------------------------------------
'''

# parse the yaml file 
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# load yaml config file with all the options you provided
config = load_config(args.config)

# creates separate directory for each experiment
exp_dir = set_up_experiment(config)

# load data
data_path = join(config['data']['data_dir'], config['data']['file_name'])
X, y = load_data(data_path)

# initialize scaler
x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']
x_scaler_params = config['preprocessing']['x_scaler_params']
y_scaler_params = config['preprocessing']['y_scaler_params']

X_scaler = StandardScaler()

# initialize estimator
est_name = config["estimator"]["name"]
est_params = config["estimator"]['est_params']

'''
--- NESTED CROSS VALIDATION----------------

This part is not for tuning final hyperparameters but for evaluation/estimation of the generalization error.
'''
outer_kf = KFold(n_splits=7, shuffle = True, random_state = 12)
inner_kf = KFold(n_splits=5, shuffle = True, random_state = 8)

param_grid = config["tuning"]["tuning_config"]["param_grid"]

results = []
for idx, (outer_train_index, outer_test_index) in enumerate(outer_kf.split(X)):
    print("outer split nr. ", idx)

    X_train, y_train = X.iloc[outer_train_index,:].copy(), y.iloc[outer_train_index].copy()
    X_test, y_test = X.iloc[outer_test_index,:].copy(), y.iloc[outer_test_index].copy()

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    combo_rmses = []
    for idx_2, combo in enumerate(param_combinations):
        if idx_2 % 2 == 0:
            print("param combo nr. ", idx_2, "/", len(param_combinations))
        rmses = []
        for (inner_train_index, inner_test_index) in inner_kf.split(X_train):
            model = create_model(est_name, est_params)
            model_params, selector_params, selector_estimator_params = split_sk_params(combo)
            model.set_params(**model_params)
            X_tune, y_tune  = X_train.iloc[inner_train_index, :].copy(), y_train.iloc[inner_train_index].copy()
            X_val, y_val  = X_train.iloc[inner_test_index, :].copy(), y_train.iloc[inner_test_index].copy()

            X_tune = X_scaler.fit_transform(X_tune)
            X_val = X_scaler.transform(X_val)

            if config["feature_selection"]["apply"] == True:
                selector_estimator = create_model(**config["feature_selection"]["estimator_config"]) 
                selector_estimator.set_params(**selector_estimator_params)
                selector_name = config["feature_selection"]["selector_name"]
                selector = get_feature_selector(name=selector_name, estimator=selector_estimator, params=selector_params) 
                X_tune = selector.fit_transform(X_tune, y_tune)
                X_val = selector.transform(X_val,y_val)

            X_tune_gpu, y_tune_gpu = cp.array(X_tune), cp.array(y_tune)
            X_val_gpu, y_val_gpu = cp.array(X_val), cp.array(y_val)

            if config["preprocessing"]["y_scaling"][0] == "log2":
                y_tune_gpu = log2p1(y_tune_gpu)
            
            if config["preprocessing"]["y_scaling"][1] == "standard":
                mu = cp.asarray(y_tune_gpu.mean())
                std = cp.asarray(y_tune_gpu.std(ddof=1))
                y_tune_gpu = (y_tune_gpu - mu) / std

            model.fit(X_tune_gpu, y_tune_gpu)
            
            y_pred = cp.asarray(model.predict(X_val_gpu))

            # scale back to original scale for validation
            if config["preprocessing"]["y_scaling"][1] == "standard":
                y_pred = y_pred*std + mu
            
            if config["preprocessing"]["y_scaling"][0] == "log2":
                y_pred = exp2m1(y_pred)
            
            y_pred_raw=y_pred.get()
            y_val_raw=y_val

            rmse = root_mean_squared_error(y_pred_raw, y_val_raw)
            rmses.append(rmse)
            del model, y_pred
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        mean_rmse = np.array(rmses).mean()
        new_row = {**combo, **{"RMSE": mean_rmse}}
        combo_rmses.append(new_row)
        
    best_params_with_rmse = min(combo_rmses, key=lambda x: x['RMSE'])
    best_params = best_params_with_rmse.copy()
    del best_params['RMSE']

    best_model_params, best_selector_params, best_selector_estimator_params = split_sk_params(best_params)
    model = create_model(est_name, est_params)
    model.set_params(**best_model_params)

    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    if config["feature_selection"]["apply"] == True:
        selector_estimator = create_model(**config["feature_selection"]["estimator_config"]) 
        selector_estimator.set_params(**best_selector_estimator_params)
        selector_name = config["feature_selection"]["selector_name"]
        selector = get_feature_selector(name=selector_name, estimator=selector_estimator, params=best_selector_params) 
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test, y_test)

    X_train_gpu,  y_train_gpu= cp.array(X_train), cp.array(y_train)
    X_test_gpu, y_test_gpu = cp.array(X_test), cp.array(y_test)

    if config["preprocessing"]["y_scaling"][0] == "log2":
        y_train_gpu = log2p1(y_train_gpu)
            
    if config["preprocessing"]["y_scaling"][1] == "standard":
        mu = cp.asarray(y_train_gpu.mean())
        std = cp.asarray(y_train_gpu.std(ddof=1))
        y_train_gpu = (y_train_gpu - mu) / std

    model.fit(X_train_gpu, y_train_gpu)
    y_pred = cp.asarray(model.predict(X_test_gpu))

    if config["preprocessing"]["y_scaling"][1] == "standard":
        y_pred = y_pred*std + mu
    
    if config["preprocessing"]["y_scaling"][0] == "log2":
        y_pred = exp2m1(y_pred)

    y_pred_raw=y_pred.get()
    y_test_raw=y_test.ravel()

    rmse = root_mean_squared_error(y_test_raw, y_pred_raw)
    r_val, _ = pearsonr(y_test_raw, y_pred_raw)

    results.append({"best_combo": str(best_params), "RMSE": rmse, "pearsonr": r_val})
    cp.get_default_memory_pool().free_all_blocks()

df = pd.DataFrame(results)
df.to_csv(join(exp_dir, "nested_crossval_results.csv"), sep='\t')

mean = df[["RMSE", "pearsonr"]].mean()
std = df[["RMSE", "pearsonr"]].std()

total_accuracy = pd.DataFrame({"mean": mean, "std": std})
total_accuracy.to_csv(join(exp_dir, "nested_cross_val_accuracy.csv"), sep='\t')

'''
--- FINAL TUNING OF HYPERPARAMETERS ON ALL DATA------------------------------
'''
print("final tuning ...")
kf = inner_kf = KFold(n_splits=5, shuffle = True, random_state = 9)
combo_rmses = []
for combo in param_combinations:
    model_params, selector_params, selector_estimator_params = split_sk_params(combo)
    rmses = []
    for (train_index, test_index) in kf.split(X):
        model = create_model(est_name, est_params)
        model.set_params(**model_params)

        X_tune, y_tune  = X.iloc[train_index, :].copy(), y.iloc[train_index].copy()
        X_val, y_val  = X.iloc[test_index, :].copy(), y.iloc[test_index].copy()

        X_tune = X_scaler.fit_transform(X_tune)
        X_val = X_scaler.transform(X_val)

        if config["feature_selection"]["apply"] == True:
            selector_estimator = create_model(**config["feature_selection"]["estimator_config"]) 
            selector_estimator.set_params(**selector_estimator_params)
            selector_name = config["feature_selection"]["selector_name"]
            selector = get_feature_selector(name=selector_name, estimator=selector_estimator, params=selector_params) 
            X_tune = selector.fit_transform(X_tune, y_tune)
            X_val = selector.transform(X_val, y_val)

        X_tune_gpu, y_tune_gpu = cp.array(X_tune), cp.array(y_tune)
        X_val_gpu, y_val_gpu = cp.array(X_val), cp.array(y_val)

        if config["preprocessing"]["y_scaling"][0] == "log2":
            y_tune_gpu = log2p1(y_tune_gpu)
        
        if config["preprocessing"]["y_scaling"][1] == "standard":
            mu = cp.asarray(y_tune_gpu.mean())
            std = cp.asarray(y_tune_gpu.std(ddof=1))
            y_tune_gpu = (y_tune_gpu - mu) / std

        model.fit(X_tune_gpu, y_tune_gpu)
        
        y_pred = cp.asarray(model.predict(X_val_gpu))

        # scale back to original scale for validation
        if config["preprocessing"]["y_scaling"][1] == "standard":
            y_pred = y_pred*std + mu
        
        if config["preprocessing"]["y_scaling"][0] == "log2":
            y_pred = exp2m1(y_pred)
        
        y_pred_raw=y_pred.get()
        y_val_raw=y_val.to_numpy()

        rmse = root_mean_squared_error(y_pred_raw, y_val_raw)
        rmses.append(rmse)
        del model, y_pred
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    mean_rmse = np.array(rmses).mean()
    new_row = {**combo, **{"RMSE": mean_rmse}}
    combo_rmses.append(new_row)


overall_best_params_with_rmse = min(combo_rmses, key=lambda x: x['RMSE'])
overall_best_params = overall_best_params_with_rmse.copy()
del overall_best_params['RMSE'] # Corrected variable name

# 1. Prepare ALL data with the BEST preprocessing parameters
model_params, selector_params, selector_estimator_params = split_sk_params(overall_best_params)

# Final Scaling
final_scaler = X_scaler # This uses the X_scaler class you defined earlier
X_full_scaled = final_scaler.fit_transform(X)

# 2. Final Y-Scaling
y_full_gpu = cp.array(y)
y_mu, y_std = 0, 1
if config["preprocessing"]["y_scaling"][0] == "log2":
    y_full_gpu = cp.log2(y_full_gpu + 1)
if config["preprocessing"]["y_scaling"][1] == "standard":
    y_mu = cp.asarray(y_full_gpu.mean())
    y_std = cp.asarray(y_full_gpu.std(ddof=1) + 1e-10)
    y_full_gpu = (y_full_gpu - y_mu) / y_std

# Final Feature Aggregation (CPU)
if config["feature_selection"]["apply"]:
    selector_estimator = create_model(**config["feature_selection"]["estimator_config"]) 
    selector_estimator.set_params(**selector_estimator_params)
    selector = get_feature_selector(name=config["feature_selection"]["selector_name"], 
                                   estimator=selector_estimator, 
                                   params=selector_params)
    X_full_final = selector.fit_transform(X_full_scaled, y_full_gpu)
    final_labels = selector.estimator.labels_
else:
    X_full_final = X_full_scaled
    final_labels = None

# 3. Final Model Fit
model = create_model(est_name, est_params)
model.set_params(**model_params)
model.fit(cp.array(X_full_final), y_full_gpu)

# 4. SAVE EVERYTHING (The "Inference Suite")
# You need all 4 pieces to make a prediction in the future
inference_suite = {
    "model": model,
    "scaler": final_scaler,
    "labels": final_labels,
    "y_stats": {"mu": float(y_mu), "std": float(y_std)},
    "config": config["preprocessing"]["y_scaling"]
}

joblib.dump(inference_suite, join(exp_dir, "inference_suite.joblib"))

with open(join(exp_dir, "tuned_params.json"), "w") as f:
    json.dump(overall_best_params, f, indent=2)
