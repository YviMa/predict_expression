from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

MODEL_REGISTRY = {
    "elastic_net": ElasticNet,
    "gradient_boost": GradientBoostingRegressor,
}

def create_model(name, params=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    if params == None:
        params = {}
    return MODEL_REGISTRY[name](**params)
