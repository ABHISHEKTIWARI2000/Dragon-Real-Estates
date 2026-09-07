from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Dragon_real_estates.joblib"
DATA_PATH = BASE_DIR / "data.csv"

training_data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, _ = next(splitter.split(training_data, training_data[:, 3]))
preprocessing_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)
preprocessing_pipeline.fit(training_data[train_indices, :13])
model = joblib.load(MODEL_PATH)

FEATURES = [
    ("CRIM", "Crime rate", "Per-capita crime rate by town", "0.00632"),
    ("ZN", "Residential zoning", "Land zoned for lots over 25k sq. ft.", "18"),
    ("INDUS", "Business land", "Non-retail business acres", "2.31"),
    ("CHAS", "River access", "1 if the property borders Charles River", "0"),
    ("NOX", "Air quality", "Nitric oxide concentration", "0.538"),
    ("RM", "Room count", "Average rooms per dwelling", "6.575"),
    ("AGE", "Property age", "Homes built before 1940", "65.2"),
    ("DIS", "City access", "Distance to employment centers", "4.09"),
    ("RAD", "Highway access", "Accessibility to radial highways", "1"),
    ("TAX", "Tax rate", "Property tax per $10,000", "296"),
    ("PTRATIO", "School ratio", "Pupil-teacher ratio by town", "15.3"),
    ("B", "Community index", "Proportion of Black residents by town", "396.9"),
    ("LSTAT", "Area profile", "Lower-status population percentage", "4.98"),
]

FEATURE_FIELDS = "".join(
    f"""<label class="field" for="feature_{index}">
      <span class="field-top"><span class="field-code">{code}</span><span class="field-name">{name}</span></span>
      <span class="field-help">{help_text}</span>
      <input id="feature_{index}" name="feature_{index}" type="number" step="any" value="{value}" required />
    </label>"""
    for index, (code, name, help_text, value) in enumerate(FEATURES, start=1)
)

HTML_FORM = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="theme-color" content="#0d2b2a" />
  <title>Dragon | Property Intelligence</title>
  <style>
    :root {{
      --ink: #102a2a; --muted: #657a79; --line: #d9e5e1; --paper: #fbfcfa;
      --teal: #0d5c57; --teal-dark: #0a403e; --mint: #dcefeb; --gold: #c4933b;
      --shadow: 0 24px 70px rgba(16, 42, 42, .14);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; color: var(--ink); background: #eef4f1; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; }}
    .shell {{ width: min(1180px, calc(100% - 32px)); margin: 32px auto; background: var(--paper); border-radius: 28px; overflow: hidden; box-shadow: var(--shadow); display: grid; grid-template-columns: 310px 1fr; min-height: 720px; }}
    .rail {{ position: relative; padding: 34px 28px; color: #eaf6f1; background: linear-gradient(155deg, #0d3d3a 0%, #092726 100%); overflow: hidden; }}
    .rail::after {{ content: ""; position: absolute; width: 270px; height: 270px; right: -145px; bottom: -120px; border: 1px solid rgba(207, 236, 225, .2); border-radius: 50%; box-shadow: 0 0 0 28px rgba(207, 236, 225, .05), 0 0 0 58px rgba(207, 236, 225, .035); }}
    .brand {{ display: flex; align-items: center; gap: 11px; font-weight: 750; letter-spacing: .02em; }}
    .brand-mark {{ width: 34px; height: 34px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,.3); border-radius: 10px; color: #e6c57f; font-size: 15px; }}
    .rail-copy {{ position: relative; z-index: 1; margin-top: 106px; }}
    .eyebrow {{ color: #e1bb70; text-transform: uppercase; letter-spacing: .16em; font-size: 11px; font-weight: 750; }}
    h1 {{ margin: 12px 0 18px; max-width: 240px; font-size: clamp(34px, 4vw, 47px); line-height: 1.03; letter-spacing: -.055em; }}
    .rail-copy p {{ color: #b7d0ca; font-size: 14px; max-width: 230px; }}
    .trust {{ position: absolute; z-index: 1; left: 28px; right: 28px; bottom: 34px; padding-top: 18px; border-top: 1px solid rgba(255,255,255,.16); color: #b7d0ca; font-size: 12px; }}
    .trust strong {{ display: block; color: #eef9f4; margin-bottom: 4px; font-size: 13px; }}
    .content {{ padding: 42px clamp(25px, 5vw, 64px) 46px; }}
    .content-header {{ display: flex; align-items: end; justify-content: space-between; gap: 20px; padding-bottom: 27px; border-bottom: 1px solid var(--line); }}
    h2 {{ margin: 0 0 5px; font-size: 25px; letter-spacing: -.035em; }}
    .intro {{ margin: 0; color: var(--muted); font-size: 14px; }}
    .step {{ color: var(--teal); font-size: 12px; font-weight: 750; white-space: nowrap; }}
    .form {{ margin-top: 30px; }}
    .form-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 22px 18px; }}
    .field {{ min-width: 0; }}
    .field-top {{ display: flex; align-items: baseline; gap: 8px; }}
    .field-code {{ font-size: 12px; font-weight: 800; letter-spacing: .09em; color: var(--teal); }}
    .field-name {{ color: var(--ink); font-size: 13px; font-weight: 700; }}
    .field-help {{ display: block; min-height: 35px; margin: 4px 0 8px; color: var(--muted); font-size: 11px; line-height: 1.35; }}
    input {{ width: 100%; padding: 13px 14px; border: 1px solid var(--line); border-radius: 10px; color: var(--ink); background: #fff; font: inherit; font-size: 14px; outline: none; transition: border-color .2s, box-shadow .2s, transform .2s; }}
    input:hover {{ border-color: #a4c4bc; }} input:focus {{ border-color: var(--teal); box-shadow: 0 0 0 4px rgba(13, 92, 87, .11); transform: translateY(-1px); }}
    .actions {{ display: flex; align-items: center; gap: 18px; margin-top: 34px; padding-top: 26px; border-top: 1px solid var(--line); }}
    button {{ display: inline-flex; align-items: center; justify-content: center; gap: 10px; min-width: 190px; padding: 14px 22px; border: 0; border-radius: 10px; color: #fff; background: var(--teal); font: inherit; font-size: 14px; font-weight: 750; cursor: pointer; box-shadow: 0 10px 24px rgba(13, 92, 87, .2); transition: transform .2s, background .2s, box-shadow .2s; }}
    button:hover {{ background: var(--teal-dark); transform: translateY(-2px); box-shadow: 0 14px 27px rgba(13, 92, 87, .27); }} button:disabled {{ opacity: .7; cursor: wait; transform: none; }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .result {{ display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-top: 26px; padding: 18px 20px; border: 1px solid #c7e3da; border-radius: 13px; color: var(--teal-dark); background: var(--mint); }}
    .result-label {{ font-size: 12px; font-weight: 750; text-transform: uppercase; letter-spacing: .1em; }} .result-value {{ margin-top: 3px; font-size: 25px; font-weight: 800; letter-spacing: -.03em; }}
    .result-mark {{ width: 42px; height: 42px; display: grid; place-items: center; border-radius: 50%; color: var(--teal); background: #f5fffb; font-size: 20px; }}
    .result.error {{ color: #8c3028; background: #fff0ee; border-color: #f2c6c0; }}
    @media (max-width: 850px) {{ .shell {{ grid-template-columns: 1fr; margin: 16px auto; }} .rail {{ min-height: 0; }} .rail-copy {{ margin-top: 62px; }} .trust {{ position: relative; left: auto; right: auto; bottom: auto; margin-top: 48px; }} .form-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 520px) {{ .shell {{ width: calc(100% - 20px); border-radius: 20px; }} .rail, .content {{ padding: 25px 20px; }} .rail-copy {{ margin-top: 68px; }} .trust {{ margin-top: 42px; }} .content-header, .actions, .result {{ align-items: flex-start; flex-direction: column; }} .form-grid {{ grid-template-columns: 1fr; gap: 18px; }} .field-help {{ min-height: 0; }} button {{ width: 100%; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <aside class="rail">
      <div class="brand"><span class="brand-mark">DR</span><span>Dragon Real Estates</span></div>
      <div class="rail-copy"><div class="eyebrow">Property intelligence</div><h1>Know the value before you move.</h1><p>Build a clearer picture of a property's potential with a data-led estimate in seconds.</p></div>
      <div class="trust"><strong>Private by design</strong>Your details stay in this session and are only used to calculate the estimate.</div>
    </aside>
    <section class="content">
      <header class="content-header"><div><h2>Property profile</h2><p class="intro">Tell us about the location and the home you are evaluating.</p></div><span class="step">01 / 01</span></header>
      <form class="form" id="predictForm"><div class="form-grid">{FEATURE_FIELDS}</div><div class="actions"><button id="submitButton" type="submit"><span>Generate estimate</span><span aria-hidden="true">→</span></button><span class="hint">13 signals · Instant analysis</span></div></form>
      <div id="result" class="result" role="status" aria-live="polite"><div><div class="result-label">Your estimate</div><div class="result-value">Ready when you are</div></div><div class="result-mark" aria-hidden="true">↗</div></div>
    </section>
  </main>
  <script>
    const form = document.getElementById('predictForm');
    const resultBox = document.getElementById('result');
    const submitButton = document.getElementById('submitButton');
    const resultValue = resultBox.querySelector('.result-value');
    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const features = Array.from(form.querySelectorAll('input')).map((field) => Number(field.value));
      resultBox.classList.remove('error');
      resultValue.textContent = 'Calculating your estimate…';
      submitButton.disabled = true;
      try {{
        const response = await fetch('/', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ features }}) }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'We could not calculate the estimate.');
        resultValue.textContent = Number(data.prediction).toFixed(2);
      }} catch (error) {{
        resultBox.classList.add('error');
        resultValue.textContent = error.message;
      }} finally {{ submitButton.disabled = false; }}
    }});
  </script>
</body>
</html>"""


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return make_prediction()
    return render_template_string(HTML_FORM)


@app.route("/api/predict", methods=["POST"])
def predict():
    return make_prediction()


def make_prediction():
    data = request.get_json(silent=True) or {}
    features = data.get("features")
    if features is None:
        return jsonify({"error": "Expected JSON body with a 'features' array."}), 400
    try:
        features_array = np.asarray(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "'features' must be a list of numeric values."}), 400
    if features_array.shape[1] != 13 or not np.isfinite(features_array).all():
        return jsonify({"error": "Please enter exactly 13 valid numeric values."}), 400
    prepared_features = preprocessing_pipeline.transform(features_array)
    prediction = model.predict(prepared_features)
    return jsonify({"prediction": float(prediction[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
