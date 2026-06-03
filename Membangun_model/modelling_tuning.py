import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_preprocessed_data(data_dir="Membangun_model/california_housing_preprocessed"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_preprocessed.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_preprocessed.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")  
    mlflow.set_experiment("California_Housing_Skilled")

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    base_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

    with mlflow.start_run(run_name="RandomForest_Tuning"):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model, "best_model")

        feature_importance = best_model.feature_importances_
        features = X_train.columns.tolist()
        plt.figure(figsize=(10,6))
        plt.barh(features, feature_importance)
        plt.xlabel("Feature Importance")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        os.remove("feature_importance.png")

        print(f"Best params: {best_params}")
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    main()