import xgboost as xgb
import optuna
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

class XGBoostModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.model = None

    def optimize_hyperparameters(self):
        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "booster": "gbtree",
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
                "tree_method": "gpu_hist",
                "gpu_id": 0,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_val)
            logloss = log_loss(self.y_val, y_pred)
            return logloss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        self.best_params = study.best_params

    def train_model(self):
        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(self.X_train, self.y_train)