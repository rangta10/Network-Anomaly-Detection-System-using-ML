from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import os
import io
import json
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)

# Global model state
model_state = {
    "isolation_forest": None,
    "encoder": None,
    "categorical_cols": None,
    "trained": False,
    "train_stats": {}
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/train', methods=['POST'])
def train():
    """Train the Isolation Forest model with uploaded CSV."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    contamination = float(request.form.get('contamination', 0.1))

    try:
        train_data = pd.read_csv(file)

        if 'class' not in train_data.columns:
            return jsonify({"error": "'class' column not found in training data"}), 400

        # Identify categorical columns (excluding 'class')
        categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
        if 'class' in categorical_cols:
            categorical_cols.remove('class')

        # One-Hot Encoding
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_train = pd.DataFrame(
            encoder.fit_transform(train_data[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        train_data_encoded = pd.concat(
            [train_data.drop(columns=categorical_cols), encoded_train], axis=1
        )

        X_train = train_data_encoded.drop('class', axis=1)

        # Train model
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        isolation_forest.fit(X_train)

        # Save to global state
        model_state["isolation_forest"] = isolation_forest
        model_state["encoder"] = encoder
        model_state["categorical_cols"] = categorical_cols
        model_state["trained"] = True
        model_state["train_stats"] = {
            "rows": len(train_data),
            "features": len(X_train.columns),
            "contamination": contamination,
            "class_distribution": train_data['class'].value_counts().to_dict()
        }

        return jsonify({
            "success": True,
            "message": "Model trained successfully",
            "stats": model_state["train_stats"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run anomaly detection on uploaded test CSV."""
    if not model_state["trained"]:
        return jsonify({"error": "Model not trained yet. Please upload training data first."}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        test_data = pd.read_csv(file)
        original_test = test_data.copy()

        if 'class' in test_data.columns:
            test_data.drop(columns=['class'], inplace=True)

        categorical_cols = model_state["categorical_cols"]
        encoder = model_state["encoder"]
        isolation_forest = model_state["isolation_forest"]

        # Encode test data
        encoded_test = pd.DataFrame(
            encoder.transform(test_data[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        test_data_encoded = pd.concat(
            [test_data.drop(columns=categorical_cols), encoded_test], axis=1
        )

        # Predict
        predictions = isolation_forest.predict(test_data_encoded)
        scores = isolation_forest.decision_function(test_data_encoded)

        # Map: 1 = normal, -1 = anomaly
        labels = ["anomaly" if p == -1 else "normal" for p in predictions]

        original_test['anomaly_prediction'] = predictions
        original_test['anomaly_label'] = labels
        original_test['anomaly_score'] = np.round(scores, 4)

        anomaly_count = int((predictions == -1).sum())
        normal_count = int((predictions == 1).sum())

        # Build per-row results (limit to 500 for display)
        rows = original_test.head(500).to_dict(orient='records')

        return jsonify({
            "success": True,
            "total": len(predictions),
            "anomalies": anomaly_count,
            "normal": normal_count,
            "anomaly_rate": round(anomaly_count / len(predictions) * 100, 2),
            "rows": rows,
            "columns": original_test.columns.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "trained": model_state["trained"],
        "stats": model_state["train_stats"] if model_state["trained"] else {}
    })


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("🚀 Anomaly Detection API running at http://localhost:5000")
    app.run(debug=True, port=5000)
