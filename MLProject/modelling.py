
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


mlflow.sklearn.autolog()

DATA_PATH = "MSFT-Stock_preprocessing.csv"
TARGET_COL = "Volatility_20D" 


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    print(f"[INFO] Total data       : {len(df)} baris")
    print(f"[INFO] Fitur (X) shape  : {X.shape}")
    print(f"[INFO] Target (y) shape : {y.shape}")
    print(f"[INFO] Kolom fitur      : {X.columns.tolist()}")
    print(f"[INFO] Kolom target     : {TARGET_COL}")
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    print(f"[INFO] Train size       : {X_train.shape}")
    print(f"[INFO] Test size        : {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_prediction_plot(y_true, y_pred, filepath="prediction_vs_actual.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, edgecolor="k", s=20)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--", lw=2, label="Ideal (y=x)",
    )
    plt.xlabel("Volatility 20D Aktual")
    plt.ylabel("Volatility 20D Prediksi")
    plt.title("Prediksi vs Aktual — MSFT Volatility (20-day)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close()
    return filepath

def save_feature_importance_plot(model, feature_names, filepath="feature_importance.png"):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[idx], color="steelblue")
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in idx],
        rotation=45,
        ha="right",
    )
    plt.xlabel("Fitur")
    plt.ylabel("Importance")
    plt.title("Feature Importance — Random Forest")
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close()
    return filepath

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)

    pred_plot = save_prediction_plot(y_test, y_pred)
    fi_plot = save_feature_importance_plot(model, X.columns.tolist())

    mlflow.log_artifact(pred_plot, artifact_path="plots")
    mlflow.log_artifact(fi_plot, artifact_path="plots")

    print("\n" + "=" * 50)
    print("HASIL EVALUASI MODEL (Test Set)")
    print("=" * 50)
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R²   : {r2:.4f}")
    print("=" * 50)
    print(f"\n[INFO] Run ID    : {mlflow.active_run().info.run_id}")
    print(f"[INFO] Artifak   : {pred_plot}, {fi_plot}")
    print(f"[INFO] Model     : ter-log otomatis oleh autolog")
    print("\n[INFO] Jalankan 'mlflow ui' untuk melihat hasil tracking.")


if __name__ == "__main__":
    main()
