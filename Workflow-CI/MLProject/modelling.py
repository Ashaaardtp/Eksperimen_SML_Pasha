import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings("ignore")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "california_housing_preprocessed")
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_preprocessed.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_preprocessed.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("CI_Retraining")
    
    with mlflow.start_run(run_name="CI_Run"):
        mlflow.sklearn.autolog()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()