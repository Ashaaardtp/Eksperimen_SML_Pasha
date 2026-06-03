import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os

def load_preprocessed_data(data_dir="Membangun_model/california_housing_preprocessed"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_preprocessed.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_preprocessed.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("California_Housing_Basic")
    
    with mlflow.start_run(run_name="RandomForest_Basic"):
        mlflow.sklearn.autolog()
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}")
        

        
if __name__ == "__main__":
    main()