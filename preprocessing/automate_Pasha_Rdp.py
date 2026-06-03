import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import argparse
import os

def load_data(url=None, filepath=None):
    if url:
        df = pd.read_csv(url)
    elif filepath:
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Butuh url atau filepath")
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df

def handle_missing(df):
    imputer = SimpleImputer(strategy='median')
    df_clean = df.copy()
    cols_missing = df.columns[df.isnull().any()].tolist()
    if cols_missing:
        df_clean[cols_missing] = imputer.fit_transform(df_clean[cols_missing])
        print(f"Filled missing in: {cols_missing}")
    else:
        print("No missing values")
    return df_clean

def encode_cat(df, cat_cols):
    df_enc = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[f'{col}_encoded'] = le.fit_transform(df_enc[col].astype(str))
            df_enc = df_enc.drop(col, axis=1)
            encoders[col] = le
            print(f"Encoded {col} -> {col}_encoded ({len(le.classes_)} categories)")
    return df_enc, encoders

def split_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    if numeric_cols:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        print(f"Scaled {len(numeric_cols)} numeric columns")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_data(X_train, X_test, y_train, y_test, out_dir='preprocessed_data'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    X_train.to_csv(os.path.join(out_dir, 'X_train_preprocessed.csv'), index=False)
    X_test.to_csv(os.path.join(out_dir, 'X_test_preprocessed.csv'), index=False)
    y_train.to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)
    print(f"Saved to {out_dir}")

def run_pipeline(url=None, filepath=None, target_col='median_house_value', cat_cols=None, out_dir='preprocessed_data'):
    print("=== PIPELINE START ===")
    df = load_data(url=url, filepath=filepath)
    df = handle_missing(df)
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df, encoders = encode_cat(df, cat_cols)
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test, scaler = split_scale(X, y)
    save_data(X_train, X_test, y_train, y_test, out_dir)
    print("=== PIPELINE DONE ===")
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'scaler': scaler, 'encoders': encoders}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
    parser.add_argument('--target', type=str, default='median_house_value')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    parser.add_argument('--categorical', type=str, nargs='+', default=None)
    args = parser.parse_args()
    run_pipeline(url=args.url, target_col=args.target, cat_cols=args.categorical, out_dir=args.output)

if __name__ == "__main__":
    main()