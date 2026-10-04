# Advanced Network Intrusion Detection Using Deep Learning

A web-based intrusion detection system. A 1-D Convolutional Neural Network (TensorFlow/Keras) classifies
network connection records as **normal**, **DoS**, **Probe**, **R2L** or **U2R**. A Flask backend serves a
web interface with login, dataset upload, prediction, charts, a dashboard and a SQLite detection history.

## Folder layout

```
nids_project/
├── app.py                    Flask backend (routes, auth, SQLite, prediction)
├── preprocess.py             Cleaning, missing values, encoding, normalisation, label mapping
├── model.py                  CNN architecture
├── train_model.py            Splits data, trains, evaluates, saves model + metrics
├── generate_sample_data.py   Makes synthetic NSL-KDD-style CSVs (already generated in data/)
├── requirements.txt
├── data/                     sample_train.csv, sample_upload_labeled.csv, sample_upload_unlabeled.csv
├── models/                   Created by train_model.py (model, preprocessor, metrics.json)
├── templates/  static/       Frontend: HTML, CSS, JavaScript (Chart.js for charts)
├── instance/                 SQLite database (nids.db) is created here on first run
└── uploads/                  Temporary upload area (files are deleted after analysis)
```

## Run it

Requires Python 3.9 to 3.12 (TensorFlow does not support newer versions yet).

```bash
cd nids_project
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python train_model.py            # trains the CNN on data/sample_train.csv (about a minute)
python app.py                    # then open http://127.0.0.1:5000
```

Sign in with **admin / admin123**, or create your own account. Then open *Analyse traffic* and upload
`data/sample_upload_labeled.csv` (or the unlabeled one).

## Using the real NSL-KDD dataset

1. Download `KDDTrain+.txt` (NSL-KDD) and put it in `data/`.
2. Train on it: `python train_model.py --data data/KDDTrain+.txt --epochs 30`
3. Restart `python app.py`. Upload `KDDTest+.txt` (or any NSL-KDD file) on the *Analyse traffic* page.

Files with or without a header row both work. Attack names (neptune, smurf, satan ...) are grouped into
the five classes automatically. The sample data is synthetic and only exists so the project runs out of the box,
so use real NSL-KDD for results you report.

## What each part does

| Requirement | Where |
|---|---|
| Data cleaning, missing values, encoding, normalisation, splitting | `preprocess.py`, `train_model.py` (70 / 15 / 15 stratified split) |
| CNN classifier | `model.py` |
| Accuracy, precision, recall, F1, confusion matrix | `train_model.py` saves them; shown on the *Model performance* page |
| Secure login | `app.py` (hashed passwords, sessions, CSRF tokens) |
| Dataset upload and prediction | *Analyse traffic* page |
| Result visualisation and dashboard statistics | *Results* and *Dashboard* pages |
| Detection history in SQLite | `instance/nids.db`, *History* page |

## Notes

- Change the default password after the first sign-in, or set the `NIDS_SECRET` environment variable for the session key.
- The charts load Chart.js from a CDN, so they need internet access. Everything else works offline.
- To retrain, run `python train_model.py` again and restart the app.
