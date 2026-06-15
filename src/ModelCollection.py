from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

DEFAULT_PARAMS = {
    "KNN": {'n_neighbors': 4, 'weights': "distance"},
    "XGBoost": {'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 3, 'gamma': 1.0, 
                'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_lambda': 1.0},
    "MLP": {'hidden_layer_sizes': (128,), 'activation': "relu", 'alpha': 1e-4, 'learning_rate_init': 1e-3, 
            'batch_size': 128, 'early_stopping': True, 'max_iter': 200, 'random_state': 42}
}

class ModelCollection:
    def __init__(self):
        pass


    def get(self, model_name, params=None):
        if params is None:
            if(model_name not in DEFAULT_PARAMS):
                params = {}
            else: params = DEFAULT_PARAMS[model_name] 

        if(model_name == "KNN"):
            return KNeighborsClassifier(**params)
        
        elif(model_name == "XGBoost"):
            params = params.copy()
            params["tree_method"] = "hist"
            return XGBClassifier(**params)
        
        elif(model_name == "MLP"):
            return MLPClassifier(**params)
        
        raise ValueError(f"Unknown model: {model_name}")