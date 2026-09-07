from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Dragon_real_estates.joblib"

# Load trained model from the project root so it works in local runs and deployment
model = joblib.load(MODEL_PATH)

HTML_FORM = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dragon Real Estates Predictor</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f7fb;
      margin: 0;
      padding: 30px;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    h1 {
      margin-top: 0;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }
    .field {
      display: flex;
      flex-direction: column;
    }
    label {
      font-size: 13px;
      font-weight: bold;
      margin-bottom: 6px;
    }
    input {
      padding: 10px 12px;
      border: 1px solid #cfd8e3;
      border-radius: 8px;
      font-size: 14px;
    }
    button {
      margin-top: 18px;
      padding: 12px 18px;
      font-size: 16px;
      border: none;
      border-radius: 8px;
      background: #2563eb;
      color: white;
      cursor: pointer;
    }
    button:hover {
      background: #1d4ed8;
    }
    .result {
      margin-top: 18px;
      padding: 14px 16px;
      background: #ecfdf5;
      border: 1px solid #a7f3d0;
      border-radius: 8px;
      font-size: 18px;
      color: #065f46;
      min-height: 24px;
    }
    .error { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Dragon Real Estates Predictor</h1>
    <form id="predictForm">
      <div class="grid">
        <div class="field"><label>CRIM<br><small>Per-capita crime rate by town</small></label><input type="number" step="any" value="0.00632" name="feature_1" /></div>
        <div class="field"><label>ZN<br><small>Residential land zoned for lots over 25k sq.ft.</small></label><input type="number" step="any" value="18" name="feature_2" /></div>
        <div class="field"><label>INDUS<br><small>Proportion of non-retail business acres</small></label><input type="number" step="any" value="2.31" name="feature_3" /></div>
        <div class="field"><label>CHAS<br><small>1 if borders Charles River, otherwise 0</small></label><input type="number" step="any" value="0" name="feature_4" /></div>
        <div class="field"><label>NOX<br><small>Nitric oxide concentration</small></label><input type="number" step="any" value="0.538" name="feature_5" /></div>
        <div class="field"><label>RM<br><small>Average number of rooms per dwelling</small></label><input type="number" step="any" value="6.575" name="feature_6" /></div>
        <div class="field"><label>AGE<br><small>Proportion of owner-occupied units built before 1940</small></label><input type="number" step="any" value="65.2" name="feature_7" /></div>
        <div class="field"><label>DIS<br><small>Weighted distance to employment centers</small></label><input type="number" step="any" value="4.09" name="feature_8" /></div>
        <div class="field"><label>RAD<br><small>Index of accessibility to radial highways</small></label><input type="number" step="any" value="1" name="feature_9" /></div>
        <div class="field"><label>TAX<br><small>Full property tax rate per $10,000</small></label><input type="number" step="any" value="296" name="feature_10" /></div>
        <div class="field"><label>PTRATIO<br><small>Pupil-teacher ratio by town</small></label><input type="number" step="any" value="15.3" name="feature_11" /></div>
        <div class="field"><label>B<br><small>Proportion of Black residents by town</small></label><input type="number" step="any" value="396.9" name="feature_12" /></div>
        <div class="field"><label>LSTAT<br><small>% lower-status population</small></label><input type="number" step="any" value="4.98" name="feature_13" /></div>
      </div>
      <button type="submit">Predict Price</button>
    </form>
    <div id="result" class="result">Prediction will appear here.</div>
  </div>

  <script>
    document.getElementById('predictForm').addEventListener('submit', async function (event) {
      event.preventDefault();

      const fields = Array.from(document.querySelectorAll('input[name^="feature_"]'));
      const features = fields.map((field) => Number(field.value));
      const resultBox = document.getElementById('result');

      resultBox.classList.remove('error');
      resultBox.textContent = 'Predicting...';

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Prediction failed.');
        }

        resultBox.textContent = `Estimated property value: ${data.prediction}`;
      } catch (error) {
        resultBox.classList.add('error');
        resultBox.textContent = error.message;
      }
    });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_FORM)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if features is None:
        return jsonify({"error": "Expected JSON body with a 'features' array."}), 400

    try:
        features_array = np.asarray(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "'features' must be a list of numeric values."}), 400

    if features_array.shape[1] != 13:
        return jsonify({"error": "Expected exactly 13 numeric features."}), 400

    prediction = model.predict(features_array)
    return jsonify({"prediction": float(prediction[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
