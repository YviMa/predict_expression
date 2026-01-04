import argparse
import pandas as pd
import json
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
from utils import load_config, load_data, compute_metrics, plot_feature_importances, plot_training_results
from model_registry import create_model
from feature_selection import get_feature_selector
from scaling import ScalerWrapper

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

if config["tuning"]["tune"]==True:
    
    with open(join(exp_dir, "tuned_params.json"), "r") as f:
        params = json.load(f)

    if args.override_arg is not None:
        replacement = eval(args.override_arg)
        params.update(replacement)
else:
    params = config["training"]["parameters"]

x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']

random_states=config["evaluation"]["random_states"]
n_splits = config["evaluation"]["n_splits"]
test_size = config["evaluation"]["test_size"]

metrics_df = pd.DataFrame({})

for n in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_states[n])
    
    X_scaler = ScalerWrapper(x_scalers)
    Y_scaler = ScalerWrapper(y_scalers)

    X_train = X_scaler.fit_transform(X_train)
    y_train = Y_scaler.fit_transform(y_train)

    X_test = X_scaler.transform(X_test)
    #y_test = Y_scaler.transform(y_test)

    if config["preprocessing"]["feature_selection"]["apply"]==True:
        feature_selector = get_feature_selector(config["preprocessing"]["feature_selection"])
        X_train = feature_selector.fit_transform(X_train, y_train)
        mask = feature_selector.get_support()
        X_test = X_test.iloc[:, mask]

    estimator_name = config["evaluation"]["estimator_name"]
    eval_config = config["evaluation"]["eval_config"]

    if bool(eval_config):
        eval_config.update({"params": params})
    else:
        eval_config = params

    model = create_model(estimator_name, params=eval_config)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred = Y_scaler.inverse_transform(y_pred)
    
    if not isinstance(y_test, np.ndarray):
        y_test=y_test.to_numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred=y_pred.to_numpy()

    metrics_dict = compute_metrics(y_test.ravel(), y_pred.ravel())

    col_name = "split " + str(n)
    metrics_df[col_name] = [metrics_dict["RMSE"], metrics_dict["pearsonr"]]

    
if config["evaluation"]["plot"]==True:
    plot_training_results(y_test, y_pred, exp_dir)
    plot_feature_importances(model, exp_dir)

metrics_df.index = ["RMSE", "pearsonr"]
metrics_df["mean"]=metrics_df.mean(axis=1)

metrics_df.to_csv(join(exp_dir, "evaluation.csv"), sep='\t')

    
