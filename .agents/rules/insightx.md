# InsightX — Workspace Rules

These rules apply to every agent, PR, and code change in this repository.
InsightX is a **chest X-ray research demonstration** — not a certified medical device.

---

## 1. Research Demo Framing (Medical Disclaimers)

- **Never** use the word **"diagnosis"** to describe model output.
  - ✅ Use: `prediction`, `finding`, `result`, `classification`, `model output`
  - ❌ Never: `diagnosis`, `diagnose`, `diagnostic result`
- Every screen or UI component that shows model output **must** include a visible disclaimer, for example:

  > ⚠️ **Research Demo Only** — This tool is not a certified medical device.
  > Results are AI predictions for research purposes only and must not be used
  > to guide clinical decisions. Always consult a qualified radiologist.

- Disclaimer text must be present in the JSX/HTML render — not hidden via CSS, not behind a click, not in a tooltip alone.

---

## 2. Data, Secrets & Repository Hygiene

### Never commit:
- Raw data files or X-ray images (any format: `.png`, `.jpg`, `.dcm`, `.npy`, etc.)
- Model weights or feature arrays **over 25 MB** (e.g., `*.pkl`, `*.h5`, `*.npy`, `*.pt`)
- `.env` files or any file containing API keys, tokens, or credentials
- Service account key files (`*.json` that contain `"type": "service_account"`)

### `.gitignore` must always include:
```gitignore
# Data & images
images_normalized/
*.png
*.jpg
*.jpeg
*.dcm

# Large ML artifacts
*.npy
*.pkl
*.h5
*.pt
*.pth
image_features_final.npy
insightx_rf_model.pkl
insightx_scaler.pkl

# Secrets
.env
.env.*
*service_account*.json
*credentials*.json
secrets/

# Python
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/

# Node / Frontend
node_modules/
build/
.cache/
```

When adding new large or sensitive files, update `.gitignore` before committing.

---

## 3. Data Splitting — Patient-Level Only

- **All** train/validation/test splits must be performed on **unique patient `uid`** values, not on individual rows or image paths.
- Frontal and Lateral X-ray views belonging to the **same `uid` must always land in the same split** — they must never be separated across train/val/test.
- Use `sklearn.model_selection.GroupShuffleSplit` or equivalent with `groups=df['uid']`.
- Document the random seed and split ratios in every notebook/script that performs splitting.

```python
# ✅ Correct — split by uid
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['uid']))

# ❌ Never do this
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2)  # rows, not patients!
```

---

## 4. X-Ray Image Privacy — No Persistence

- Uploaded X-ray images must **never be written to disk**, stored in a database, logged, or cached anywhere.
- Process all uploaded images **in memory only** (e.g., `PIL.Image`, `numpy` arrays, `io.BytesIO`).
- Backend endpoints that receive image uploads must return the result and immediately discard the data.
- Never log file names, file sizes, pixel statistics, or any metadata that could identify a patient.

---

## 5. Language, Tooling & Stack

### Python (Backend / ML)
- **Python 3.11+** — use 3.11 features where appropriate (e.g., `tomllib`, `Self`, `match`)
- **Type hints are mandatory** on all function signatures and class attributes
- **Linter:** `ruff` — run `ruff check .` before every commit; zero warnings policy
- **Tests:** `pytest` — new logic must have corresponding unit tests; run `pytest` before merging
- **Formatter:** `ruff format .` (or `black` as fallback if ruff format unavailable)

### Frontend
- **React with Vite** — do not use Create React App (`react-scripts`) for new work; migrate existing CRA setup to Vite
- Keep component files co-located with their CSS modules
- Use `lucide-react` for icons, `recharts` for charts, `framer-motion` for animations (already in `package.json`)

---

## 6. Planning & Change Transparency

- **Before editing 3 or more files in a single task**, show a short numbered plan and wait for confirmation (or note it clearly if proceeding autonomously).
- **After completing any task**, provide a summary in this format:

  ```
  ### ✅ What Changed
  - [file]: [what was changed and why]

  ### 🔍 How to Verify
  - [step 1]
  - [step 2]
  ```

- Non-trivial changes to the ML pipeline or data processing must include a note on how results could differ from a previous run.

---

## 7. Secret Hygiene

- **Never** `print()`, `logging.info()`, `console.log()`, or otherwise output:
  - API keys, tokens, or passwords
  - File paths containing usernames or sensitive directory names
  - Patient identifiers (`uid`, names, dates)
- If a secret is accidentally printed during debugging, remove the log line before committing — do not just comment it out.
- Use environment variables for all credentials; access them via `os.environ.get("KEY")` — never hardcode values.

---

*Last updated: 2026-10-11 | Project: InsightX v1.0.0*
