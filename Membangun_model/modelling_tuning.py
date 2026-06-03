import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import dagshub  

def get_data_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "california_housing_preprocessed")
    return data_dir

def load_preprocessed_data():
    data_dir = get_data_dir()
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_preprocessed.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_preprocessed.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_preprocessed_data()

dagshub.init(repo_owner='Ashaaardtp', repo_name='Eksperimen_SML_Pasha', mlflow=True)

mlflow.set_experiment("California_Housing_Advanced")

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

base_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error')

with mlflow.start_run(run_name="RandomForest_Advanced_DagsHub"):
    print("Menjalankan GridSearch...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Parameter Terbaik: {best_params}")

    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(best_model, "best_model")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")


    feature_importance = best_model.feature_importances_
    features = X_train.columns.tolist()
    plt.figure(figsize=(10,6))
    plt.barh(features, feature_importance)
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance Plot")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    residuals = y_test - y_pred
    plt.figure(figsize=(10,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig("residual_plot.png")
    mlflow.log_artifact("residual_plot.png")

    os.remove("feature_importance.png")
    os.remove("residual_plot.png")

    print("Proses selesai")