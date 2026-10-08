# Phishing URL Detector (Streamlit)

Paste a URL and a Random Forest model (trained on URL-only features) estimates the phishing probability.

## Files
- `app.py` - Streamlit app
- `features.py` - URL feature extraction (same code used in training)
- `phishing_random_forest_model.pkl` - trained model
- `requirements.txt` - pinned dependencies (scikit-learn must match the training version)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push these files to GitHub, then create the app at https://share.streamlit.io
(Advanced settings -> Python 3.12).
