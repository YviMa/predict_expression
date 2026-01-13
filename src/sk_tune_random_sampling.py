import argparse
import json
import joblib
import pandas as pd
from scaling import build_scaler
from os.path import join
from sklearn.model_selection import KFold
from utils import load_config, set_up_experiment, load_data, setup_param_dist
from model_registry import create_model
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from feature_selection import get_feature_selector

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

y = y.values.ravel()

# initialize scaler
x_scalers = config['preprocessing']['x_scaling']
y_scalers = config['preprocessing']['y_scaling']
x_scaler_params = config['preprocessing']['x_scaler_params']
y_scaler_params = config['preprocessing']['y_scaler_params']

X_scaler = build_scaler(x_scalers, x_scaler_params)
Y_scaler = build_scaler(y_scalers, y_scaler_params)

# initialize estimator
est_name = config["estimator"]["name"]
est_params = config["estimator"]['est_params']
estimator = create_model(est_name, est_params)

# build pipeline
steps = [('scaler', X_scaler)]
if config["feature_selection"]["apply"] == True:
    # initialize feature_selector
    selector_estimator = create_model(**config["feature_selection"]["estimator_config"]) # FIXED params for the estimator
    selector_name = config["feature_selection"]["selector_name"]
    selector_params = config["feature_selection"]["selector_params"]
    selector = get_feature_selector(name=selector_name, estimator=selector_estimator, params=selector_params) # params for the selector e.g. importance_getter
    steps.append(('selector', selector))

steps.append(('regressor', estimator))

pipe = Pipeline(steps)

model = TransformedTargetRegressor(
    regressor=pipe,
    transformer=Y_scaler
)
'''
--- NESTED CROSS VALIDATION----------------

This part is not for tuning final hyperparameters but for evaluation/estimation of the generalization error.
'''
# initialize outer cross-validation folds
kf = KFold(n_splits=7, shuffle = True, random_state =42)

tuning_config = config["tuning"]["tuning_config"]
param_dist = setup_param_dist(tuning_config["param_distributions"])
tuning_config["param_distributions"] = param_dist

results = []
# iterating over the outer folds
for idx, (train_index, test_index) in enumerate(kf.split(X)):
    print("currently running split nr. ", idx)
    X_train, y_train = X.iloc[train_index,:], y[train_index]
    X_test, y_test = X.iloc[test_index,:], y[test_index]

    grid = RandomizedSearchCV(model, **tuning_config, verbose =3, random_state=7)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    y_pred = grid.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r_val, _ = pearsonr(y_test, y_pred.ravel())

    new_row = {"best_combo": str(best_params), "RMSE":rmse, "pearsonr": r_val}
    results.append(new_row)

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

grid = RandomizedSearchCV(model, **tuning_config, random_state =7)
grid.fit(X, y)
overall_best_params = grid.best_params_
overall_best_model = grid.best_estimator_

with open(join(exp_dir,"tuned_params.json"), "w") as f:
    json.dump(overall_best_params, f, indent=2)

print("saved best params")
# saving the model
model_filename = join(exp_dir, "model.joblib")
joblib.dump(overall_best_model, model_filename)
