import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from data_pipeline import download_data, load_and_clean
from forecaster import engineer_features, train_model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model using test data."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean(np.square(y_test - predictions)))
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:.4f}")
    print(f"R² score: {r2:.4f}")

    return predictions, rmse, r2


if __name__ == "__main__":
    download_data()
    cleaned_df = load_and_clean()
    engineered_df = engineer_features(cleaned_df)
    model, X_test, y_test = train_model(engineered_df)
    predictions, rmse, r2 = evaluate_model(model, X_test, y_test)

    print("Model evaluation complete")
    print(f"Predictions: {len(predictions)} values, RMSE={rmse:.4f}, R²={r2:.4f}")
