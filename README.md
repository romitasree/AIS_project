# terratrace

### API key setup

Use either environment variables or Streamlit secrets:

```powershell
$env:GOOGLE_API_KEY="your_real_key_here"
```

Or add `.streamlit/secrets.toml`:

# TerraTrace

> TerraTrace is a toolkit for analyzing and forecasting land-cover change using Google Earth Engine (DynamicWorld) plus a local LSTM forecasting pipeline. It includes data extraction helpers, preprocessing, model training, and a Streamlit-based UI for visualizing predictions and regional change.

## Key features

- Extract DynamicWorld land-cover summaries from Google Earth Engine for sample locations.
- Compute regional land-cover change.
- Train an LSTM forecast model from preprocessed windows and produce multi-year forecasts.
- Streamlit app and Folium maps for quick visual inspection of predictions and change tiles.

## Repository layout

- `modules/` — main Python modules (extraction, preprocessing, training, Streamlit app, analysis).
	- `app.py` — Streamlit application and mapping helpers.
	- `gee_extractor.py`, `gee_fetch.py` — Earth Engine extraction utilities.
	- `preprocess.py` — preprocessing pipeline that builds training windows.
	- `train.py` — model training script (LSTM) that writes `outputs/model.keras`.
	- `gemini_integration.py` — utilities for (local) Gemini / Romita diffs (dummy data included).
	- `policy.py`, `analyzer.py` — analysis, scoring and policy-brief helpers.
- `pixel_prediction/` — pixel-level pipeline variants and artifacts.
- `outputs/` and `pixel_output/` — generated artifacts (models, cleaned CSVs, windows.npz, reports).
- `data/` and `assets/` — small sample inputs (e.g., `data/sample.csv`, `assets/dynamicworld.json`).
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.9+ is recommended. See `requirements.txt` for exact dependency versions.
- Google Earth Engine (GEE) account and the `earthengine-api` Python package.
- Optional: TensorFlow for model training / inference (GPU if you want faster training).

Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows with Git Bash / WSL use: source .venv/Scripts/activate
pip install -r requirements.txt
```

Note: On Windows you may prefer PowerShell or WSL; the project has been used with WSL / Git Bash.

## Authentication and credentials

- Google Earth Engine: run `python -c "import ee; ee.Authenticate(); ee.Initialize()"` or follow the `ee.Authenticate()` prompt when running the Streamlit app. You can also use a service account and set `GOOGLE_APPLICATION_CREDENTIALS` to the JSON key file.
- Do not hardcode API keys. The repo includes a short note about `GOOGLE_API_KEY` usage in `modules/gemini_integration.py` (dummy/local only).

## Quick start — try it locally

1. Install dependencies and authenticate with Earth Engine (see above).

2. Run the Streamlit UI (recommended for visual exploration):

```bash
streamlit run modules/app.py
```

3. Run the extraction + preprocessing + training pipeline (non-interactive):

```bash
python modules/gee_extractor.py   # pulls DynamicWorld summaries from GEE and writes outputs/
python modules/preprocess.py     # builds windows.npz and cleaned CSVs in outputs/
python modules/train.py          # trains LSTM, writes outputs/model.keras and outputs/loss_curve.png
```

4. Run the sample landcover analysis script (CLI):

```bash
python modules/gee_fetch.py
```

Notes:
- Many scripts assume `outputs/` or `pixel_output/` exist and will look there for artifacts (`model.keras`, `scaler.pkl`, `windows.npz`).
- If you want to run the pixel-level pipeline, check `pixel_prediction/` for the specialized extractor and training steps.

## Data & artifacts

- `outputs/windows.npz` — training windows used by the LSTM.
- `outputs/model.keras` — trained model (Keras native format).
- `outputs/scaler.pkl` — input scaler used by the prediction pipeline.
- `outputs/cleaned/` — cleaned CSVs for each sample region.

If you use the Streamlit app, it will attempt to load models from `pixel_output/` first, then fall back to `outputs/`.

## Common issues & troubleshooting

- Missing GEE credentials / Authentication errors: make sure `ee.Authenticate()` completes and you have an active Earth Engine account.
- Missing artifacts (model/scaler): run `python modules/train.py` or copy a trained `model.keras` and `scaler.pkl` into `outputs/` or `pixel_output/`.
- TensorFlow issues on Windows: consider using WSL or a conda environment, and match CUDA/cuDNN versions if you plan to use GPU.
- If `st.session_state` tile helpers return None: GEE map generation sometimes fails when offline or if the request limit is hit; ensure network access and valid GEE initialization.

## Development notes

- The code uses a lightweight in-repo module layout (`modules/`). When importing in scripts, Python's import path is adjusted (see `modules/app.py`). Running modules from the repository root with `python modules/<script>.py` works reliably.
- Add new extraction points by updating `modules/gee_fetch.py` or `data/sample.csv` and re-running the extractor.

## Contributing

1. Open an issue describing the feature or bug.
2. Create a branch, implement tests where possible, and open a PR.

## License

This repository does not include an explicit license file. Add a LICENSE if you want to make the code reusable under a permissive or restrictive license.

## Where to go next

- Inspect `modules/app.py` to understand the UI flow and map helpers.
- Explore `modules/train.py` to tweak model architecture, training hyperparameters, or dataset splits.
- If you need help setting up Earth Engine or TensorFlow, mention your OS and Python environment in an issue and include error logs.

---

If you'd like, I can also add a minimal `CONTRIBUTING.md`, a sample `env.example`, or CI configuration to run a simple lint/test step. Tell me which you'd prefer next.
