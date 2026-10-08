# Phishing Detector (Streamlit)

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (app.py, features.py, requirements.txt, the .pkl).
2. Go to https://share.streamlit.io -> New app -> pick the repo, branch, and `app.py`.
3. Deploy.

Keep `scikit-learn==1.6.1` pinned: the model was pickled with that version.
