from flask import Flask, render_template, jsonify
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import download_data, load_and_clean
from forecaster import engineer_features, train_model
from metrics import evaluate_model

app = Flask(__name__)

# ── Run pipeline once at startup ──────────────────────────────────────────────
print("Initializing Smart Grid AI pipeline…")
download_data()
_cleaned     = load_and_clean()
_engineered  = engineer_features(_cleaned)
_model, _X_test, _y_test = train_model(_engineered)
_predictions, _rmse, _r2 = evaluate_model(_model, _X_test, _y_test)
print(f"Pipeline ready  |  RMSE={_rmse:.2f} MW  |  R²={_r2:.4f}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    timestamps = _y_test.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    actual     = [round(float(v), 2) for v in _y_test.values]
    predicted  = [round(float(v), 2) for v in _predictions]
    load       = [round(float(v), 2)
                  for v in _X_test["DE_load_actual_entsoe_transparency"].values]

    return jsonify({
        "timestamps": timestamps,
        "actual":     actual,
        "predicted":  predicted,
        "load":       load,
        "metrics": {
            "rmse":    round(float(_rmse), 2),
            "r2":      round(float(_r2),   4),
            "n_points": len(timestamps),
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
