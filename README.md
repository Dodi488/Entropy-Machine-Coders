# Exoplanet AI Verifier — NASA Space Apps 2025

A lightweight ML system to classify **exoplanets vs. false positives** using fused **Kepler + K2** data.

- **Backend (Flask)**: `/predict` endpoint for real-time inference.
- **Frontend (HTML/JS)**: simple UI to submit features and view the model’s prediction.
- **Data script**: optional helper to fuse Kepler & K2 if you need to regenerate the dataset.

---

## 1) Tech Stack

- **Python 3.10+**
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `flask`, `flask-cors`
- (Optional) Any static host for the frontend (GitHub Pages, Netlify, Vercel)
- (Optional) Any PaaS for the backend (Render, Railway, Fly.io, etc.)

---

## 2) Repository Structure (suggested)

```
.
├─ back_end.py                # Flask API with /predict (RUN THIS)
├─ index.html                 # Frontend (UI)
├─ fused_kepler_k2_data.csv   # Fused dataset used by the model
├─ raw_data/                  # (optional) original Kepler/K2/TESS CSVs
├─ combine_keppler_k2.py      # (optional) Kepler+K2 fusion → fused_kepler_k2_data.csv
└─ requirements.txt           # Dependencies
```

> The backend **loads/trains** once at startup from `fused_kepler_k2_data.csv` and keeps the model & scaler in memory for subsequent requests.

---

## 3) Quick Start (the shortest path)

### 3.0 Get the dependencies file
If you **cloned** this repository, you already have `requirements.txt`.  
If you **downloaded individual files**, make sure to place `requirements.txt` in the project root before installing.

### 3.1 Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3.2 Install dependencies (from `requirements.txt`)

```bash
pip install -r requirements.txt
```

> Alternatively (not recommended), install manually:
> ```bash
> pip install pandas numpy scikit-learn flask flask-cors
> ```

### 3.3 Run the backend (Flask API)

```bash
python back_end.py
```

- Default URL: `http://127.0.0.1:5000`
- CORS is enabled so a static frontend can call the API from another origin.
- The model is loaded/trained **once** at startup from `fused_kepler_k2_data.csv`.

### 3.4 Open the frontend (UI)

**Mode A — Static HTML + Local Backend**
1. Open `index.html` directly in your browser.
2. Ensure the script inside `index.html` points to your backend (e.g., `http://127.0.0.1:5000/predict`).

**Mode B — Serve the HTML from Flask (optional)**
1. Create a folder `templates/` and move `index.html` to `templates/index.html`.
2. In `back_end.py`, render it with `render_template("index.html")`.
3. Visit `http://127.0.0.1:5000/`.

---

## 4) API Reference

### POST `/predict`

**Request (JSON)**

```json
{
  "orbital_period": 8.46,
  "planet_radius": 11.2,
  "stellar_temp": 5925
}
```

**Response (JSON)**

```json
{ "prediction": "PLANET" }   // or "FALSE POSITIVE"
```

**Feature order is important**:  
`["orbital_period", "planet_radius", "stellar_temp"]`

**Errors**
- `400`: Missing or non-numeric fields
- `422`: Out-of-range values (if sanity checks are enabled)
- `500`: Model initialization failure or unexpected error

---

## 5) Feature Dictionary

| Feature         | Type   | Unit        | Example | Notes                          |
|----------------|--------|-------------|---------|--------------------------------|
| orbital_period | float  | days        | 12.3    | Orbital period of candidate    |
| planet_radius  | float  | Earth radii | 2.1     | Estimated planetary radius     |
| stellar_temp   | float  | Kelvin      | 5400    | Host star effective temperature|

---

## 6) Model Card (short)

- **Algorithm:** RandomForest (100 trees, class_weight="balanced", random_state=42)  
- **Training data:** Fused Kepler + K2 (cleaned, standardized columns)  
- **Target:** Binary — PLANET vs FALSE POSITIVE (from `disposition`)  
- **Preprocessing:** MinMaxScaler for numeric features; LabelEncoder for target  
- **Intended use:** Educational/demo classifier for hackathon. Not for science-grade discovery.  
- **Limitations:** Only 3 numeric features; no uncertainty estimates; threshold not calibrated.  
- **Notes:** Outputs should not be interpreted as confirmation. Use for exploration only.

---

## 7) (Optional) Regenerate the fused dataset

You **do not** need this for running the app. Only if you want to rebuild the dataset:

1. Place original CSVs under `raw_data/`:
   - Kepler: `cumulative_YYYY.MM.DD.csv`
   - K2:     `k2pandc_YYYY.MM.DD.csv`
2. Adjust input paths inside `combine_keppler_k2.py` if needed.
3. Run:
   ```bash
   python combine_keppler_k2.py
   ```
   Output: `fused_kepler_k2_data.csv`

---

## 8) Deployment (high-level)

### Frontend (static)
- Deploy `index.html` on GitHub Pages / Netlify / Vercel.
- Edit the `backendUrl` in `index.html` to point to the **public** backend URL.

### Backend (Flask)
- Deploy on your preferred PaaS (Render/Railway/Fly.io/…).
- Ensure:
  - `fused_kepler_k2_data.csv` is available at build/run time.
  - The service exposes the correct port (often `5000` internally; use the platform’s assigned external port).
  - CORS is enabled (`flask-cors`).
  - The API endpoint remains `/predict` and the **feature order** stays the same.

## 9) Smoke Tests

### CURL
```bash
curl -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/json"   -d '{"orbital_period": 12.3, "planet_radius": 2.1, "stellar_temp": 5400}'
```

### Browser Console
```js
fetch('http://127.0.0.1:5000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    orbital_period: 12.3,
    planet_radius: 2.1,
    stellar_temp: 5400
  })
}).then(r => r.json()).then(console.log);
```

---

## 10) Troubleshooting

- **CORS / Network errors**
  - Ensure `back_end.py` is running.
  - Confirm the `backendUrl` in `index.html` matches your backend address.
- **404 on “/” in Flask**
  - Only if you render the page from Flask: move `index.html` to `templates/index.html`.
- **Column / schema errors**
  - `fused_kepler_k2_data.csv` must include `orbital_period`, `planet_radius`, `stellar_temp` (and `disposition` for training, already prepared).
- **Front-end not updating**
  - Hard-refresh, or open DevTools → Network to confirm the `/predict` call and payload/response.

---

## 11) Acknowledgments

- NASA Space Apps Challenge 2025 (Mérida)  
- Kepler/K2 catalogs (educational/demo use)  
- scikit-learn & Flask communities

---

## License

NASA Hackaton 2025, EMC (Entropy Machine Coders)
