import argparse
import pandas as pd
import json
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
from utils import load_config, load_data, compute_metrics, plot_feature_importances, plot_training_results, log2p1, exp2m1
from model_registry import create_model
from feature_selection import get_feature_selector
from scaling import build_scaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Path to YAML config")
parser.add_argument("--override_arg", required=False, help="Argument and value to be overriden")
parser.add_argument("--exp_dir", required=True, help="Path to experiment folder")
args = parser.parse_args()

exp_dir = args.exp_dir

if args.config is not None:
    config = load_config(args.config)
else:
    config = load_config(join(exp_dir, "config.yaml"))

data_path = join(config['data']['data_dir'], config['data']['file_name'])
X, y = load_data(data_path)

x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']
x_scaler_params = config['preprocessing']['x_scaler_params']
y_scaler_params = config['preprocessing']['y_scaler_params']

if config["tuning"]["tune"]==True:
    
    with open(join(exp_dir, "tuned_params.json"), "r") as f:
        params = json.load(f)

    if args.override_arg is not None:
        replacement = eval(args.override_arg)
        params.update(replacement)
else:
    params = config["training"]["parameters"]

random_state=config["evaluation"]["random_state"]
n_splits = config["evaluation"]["n_splits"]
test_size = config["evaluation"]["test_size"]

est_name = config["estimator"]["name"]
est_params = config["estimator"]['est_params']
estimator = create_model(est_name, est_params)

kf = KFold(n_splits=5, shuffle = True, random_state =42)

metrics_df = pd.DataFrame({})

for n, (train_index, test_index) in enumerate(kf.split(X)):

    X_train, y_train = X.iloc[train_index,:], y.iloc[train_index,:]
    X_test, y_test = X.iloc[test_index,:], y.iloc[test_index,:]

    X_scaler = build_scaler(x_scalers, x_scaler_params)
    Y_scaler = build_scaler(y_scalers, y_scaler_params)

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

    model.set_params(**params)

    
    if config["preprocessing"]["feature_selection"]["apply"]==True:
        feature_selector = get_feature_selector(config["preprocessing"]["feature_selection"])
        X_train = feature_selector.fit_transform(X_train, y_train)
        mask = feature_selector.get_support()
        X_test = X_test.iloc[:, mask]
    

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test = y_test.to_numpy() if isinstance(y_test, pd.DataFrame) else y_test
    y_pred = y_pred.to_numpy() if isinstance(y_pred, pd.DataFrame) else y_pred

    metrics_dict = compute_metrics(y_test.ravel(), y_pred.ravel())

    col_name = "split " + str(n)
    metrics_df[col_name] = [metrics_dict["RMSE"], metrics_dict["pearsonr"]]

  
if config["evaluation"]["plot"]==True:
    plot_training_results(y_test, y_pred, exp_dir)
    plot_feature_importances(model, exp_dir)

metrics_df.index = ["RMSE", "pearsonr"]
metrics_df["mean"]=metrics_df.mean(axis=1)

metrics_df.to_csv(join(exp_dir, "evaluation.csv"), sep='\t')

    
