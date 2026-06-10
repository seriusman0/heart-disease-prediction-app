from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    }
    base = LogisticRegression(max_iter=1000, random_state=42)
    gs = GridSearchCV(base, param_grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    base = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(base, param_grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
    }
    base = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    gs = GridSearchCV(base, param_grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_all(X_train, y_train) -> dict:
    print("  Training Logistic Regression...")
    lr, lr_params = train_logistic_regression(X_train, y_train)
    print(f"    Best params: {lr_params}")

    print("  Training Random Forest...")
    rf, rf_params = train_random_forest(X_train, y_train)
    print(f"    Best params: {rf_params}")

    print("  Training XGBoost...")
    xgb, xgb_params = train_xgboost(X_train, y_train)
    print(f"    Best params: {xgb_params}")

    return {
        "logistic_regression": lr,
        "random_forest": rf,
        "xgboost": xgb,
    }
