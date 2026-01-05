import argparse
import matplotlib.pyplot as plt
import json
import ast
import pandas as pd
from scaling import build_scaler
from os.path import join
from sklearn.model_selection import KFold
from utils import load_config, set_up_experiment, load_data, log2p1, exp2m1
from model_registry import create_model
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from feature_selection import get_feature_selector
from sklearn import set_config

#set_config(transform_output = "pandas")

# parse the yaml file 
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# load yaml config file with all the options you provided
config = load_config(args.config)

# creates separate directory for each experiment
exp_dir = set_up_experiment(config)

data_path = join(config['data']['data_dir'], config['data']['file_name'])
X, y = load_data(data_path)

x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']
x_scaler_params = config['preprocessing']['x_scaler_params']
y_scaler_params = config['preprocessing']['y_scaler_params']

X_scaler = build_scaler(x_scalers, x_scaler_params)
Y_scaler = build_scaler(y_scalers, y_scaler_params)

est_name = config["estimator"]["name"]
est_params = config["estimator"]['est_params']
estimator = create_model(est_name, est_params)


steps = [('scaler', X_scaler)]
if config["feature_selection"]["apply"] == True:
    selector = get_feature_selector(config["feature_selection"]["selector"], config["feature_selection"]["selector_config"])
    steps.append(('selector', selector))

steps.append(('regressor', estimator))

pipe = Pipeline(steps)

model = TransformedTargetRegressor(
    regressor=pipe,
    transformer=Y_scaler
)

kf = KFold(n_splits=7, shuffle = True, random_state =42)

#tuner_name = config["tuning"]["tuner"]
tuning_config = config["tuning"]["tuning_config"]

results = []
for idx, (train_index, test_index) in enumerate(kf.split(X)):

    X_train, y_train = X.iloc[train_index,:], y.iloc[train_index,:]
    X_test, y_test = X.iloc[test_index,:], y.iloc[test_index,:]

    grid = GridSearchCV(model, **tuning_config)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    y_pred = grid.predict(X_test)
    y_test_raw = y_test.values.ravel()

    rmse = root_mean_squared_error(y_test_raw, y_pred)
    r_val, _ = pearsonr(y_test_raw, y_pred.ravel())

    new_row = {"best_combo": str(best_params), "RMSE":rmse, "pearsonr": r_val}
    results.append(new_row)

df = pd.DataFrame(results)
df.to_csv(join(exp_dir, "nested_crossval_results.csv"), sep='\t')

mean = df[["RMSE", "pearsonr"]].mean()
std = df[["RMSE", "pearsonr"]].std()

total_accuracy = pd.DataFrame({"mean": mean, "std": std})
total_accuracy.to_csv(join(exp_dir, "nested_cross_val_accuracy.csv"), sep='\t')

# final tuning of hyperparameters
grid = GridSearchCV(model, **tuning_config)
grid.fit(X, y)
overall_best_params = grid.best_params_

with open(join(exp_dir,"tuned_params.json"), "w") as f:
    json.dump(overall_best_params, f, indent=2)