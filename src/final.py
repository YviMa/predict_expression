"""
Script to load trained models per gene and make predictions on test data.
Output is saved as TSV files: pred_gene_1.tsv, pred_gene_2.tsv, pred_gene_3.tsv
DISCLAIMER: Please change the paths to filenames and/or filenames if necessary!
"""

import os
import sys
import types
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import ElasticNet, Lars, LassoLars
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration


#=========================================================
#  Utils

def log2p1(X):
    return np.log2(X + 1)

def exp2m1(X):
    return np.exp2(X) - 1

utils = types.ModuleType("utils")
utils.log2p1 = log2p1
utils.exp2m1 = exp2m1
sys.modules["utils"] = utils



# Feature Selection

FEATURE_SELECTION_REGISTRY = {
    "variance_threshold": VarianceThreshold,
    "scikitlearn_estimator": SelectFromModel
}


def get_feature_selector(name, estimator=None, params=None):
    if name not in FEATURE_SELECTION_REGISTRY:
        raise ValueError(f"Unknown feature selector: {name}")

    if params is None:
        params = {}

    return FEATURE_SELECTION_REGISTRY[name](estimator, **params)


class SelectFromTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self.estimator.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.estimator.fit_transform(X, y)

    def transform(self, X, y=None):
        return self.estimator.transform(X)

    def inverse_transform(self, X, y=None):
        return self.estimator.inverse_transform(X)


FEATURE_SELECTION_REGISTRY["scikitlearn_transformation"] = SelectFromTransformation



# Model Registry


MODEL_REGISTRY = {
    "elastic_net": ElasticNet,
    "gradient_boost": GradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "support_vector": SVR,
    "lasso_lars": LassoLars,
    "svc": SVC,
    "pca": PCA
}


def create_model(name, params=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    if params is None:
        params = {}

    return MODEL_REGISTRY[name](**params)


class ClassifierGuidedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None):
        labels = (y > 0).astype(int)

        self.classifier_ = clone(self.classifier)
        self.regressor_ = clone(self.regressor)

        self.classifier_.fit(X, labels)

        mask = labels == 1

        if not np.any(mask):
            self.regressor_ = DummyRegressor(strategy="constant", constant=0.0)
            self.regressor_.fit(X, y)
            return self

        self.regressor_.fit(X[mask], y[mask])
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        labels = self.classifier_.predict(X)
        preds = np.zeros(X.shape[0])

        mask = labels == 1
        if np.any(mask):
            preds[mask] = self.regressor_.predict(X[mask])

        return preds


MODEL_REGISTRY["hierarchical"] = ClassifierGuidedRegressor


class CustomFeatureAgg(FeatureAgglomeration):
    def fit(self, X, y=None):
        N = X.shape[1]
        conn = np.tri(N)
        conn[np.diag_indices(N)] = 0
        self.connectivity = conn
        return super().fit(X, y)
    

MODEL_REGISTRY["feature_agg"] = CustomFeatureAgg

model_registry = types.ModuleType("model_registry")
model_registry.ClassifierGuidedRegressor = ClassifierGuidedRegressor
model_registry.CustomFeatureAgg = CustomFeatureAgg
sys.modules["model_registry"] = model_registry


# Scaler


class Log2Scaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return log2p1(X)

    def inverse_transform(self, X):
        return exp2m1(X)


#=========================================================
# Main Script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to test data (TSV)")
    parser.add_argument("--gene", type=int, required=True, choices=[1,2,3], help="Gene number (1,2,3)")
    parser.add_argument("--output", required=True, help="Output TSV file for predictions")
    args = parser.parse_args()

    #Load test data
    test_df = pd.read_csv(args.input, sep='\s+')
    
    #Extract IDs and features
    ids = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    model_files = {
        1: "models/model-1.joblib",
        2: "models/model-2.joblib",
        3: "models/model-3.joblib"
    }

    model_file = model_files[args.gene]
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading model: {model_file}")
    model = joblib.load(model_file)

    #Predict
    y_pred = model.predict(X_test)
    
    out_df = pd.DataFrame({"ID": ids, "Expression": y_pred})
    out_df.to_csv(args.output, sep='\t', index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
