import joblib
import pandas as pd
import streamlit as st

from features import extract_url_features

MODEL_PATH = "url_phishing_model.pkl"

st.set_page_config(page_title="Phishing URL Detector", page_icon="🎣", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
FEATURES = list(model.feature_names_in_)
PHISH_IDX = list(model.classes_).index(1)  # 1 = phishing, 0 = legitimate


def featurize(urls):
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows)[FEATURES]


def predict(urls):
    return model.predict_proba(featurize(urls))[:, PHISH_IDX]


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Phishing threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Flag a URL as phishing when its probability is at or above this value.")
st.sidebar.caption(f"Random Forest · {len(model.estimators_)} trees · {len(FEATURES)} URL features")

# ---------------- Main ----------------
st.title("🎣 Phishing URL Detector")
st.write("Paste a link and the model estimates how likely it is to be a phishing URL. "
         "It only analyses the **text of the URL**; it never visits the site.")

tab_one, tab_many = st.tabs(["Check one URL", "Check many URLs"])

with tab_one:
    url = st.text_input("URL", placeholder="https://example.com/login")
    if st.button("Check URL", type="primary") and url.strip():
        p = float(predict([url])[0])
        if p >= threshold:
            st.error(f"⚠️ Likely PHISHING: {p:.1%} probability")
        else:
            st.success(f"✅ Likely LEGITIMATE: {p:.1%} phishing probability")
        st.progress(min(max(p, 0.0), 1.0))
        with st.expander("Features extracted from this URL"):
            st.dataframe(featurize([url]).T.rename(columns={0: "value"}))

with tab_many:
    st.write("Enter one URL per line, or upload a CSV that has a `url` column.")
    text = st.text_area("URLs", height=150, placeholder="https://example.com\nhttp://suspicious-site.xyz/login")
    up = st.file_uploader("…or CSV with a 'url' column", type="csv")

    urls = [u.strip() for u in text.splitlines() if u.strip()]
    if up is not None:
        csv = pd.read_csv(up)
        if "url" not in csv.columns:
            st.error("The CSV needs a column named 'url'.")
        else:
            urls = csv["url"].dropna().astype(str).tolist()

    if st.button("Check all") and urls:
        probs = predict(urls)
        out = pd.DataFrame({
            "url": urls,
            "phishing_probability": probs.round(4),
            "prediction": ["phishing" if p >= threshold else "legitimate" for p in probs],
        })
        st.metric("Flagged as phishing", f"{(probs >= threshold).sum()} / {len(out)}")
        st.dataframe(out)
        st.download_button("Download results", out.to_csv(index=False).encode(),
                           "url_predictions.csv", "text/csv")

st.divider()
st.caption(
    "⚠️ Limitations: this model uses URL text only (about 90% accuracy on the "
    "test split of the Hannousse & Yahiouche 2020 dataset). It can misjudge short, "
    "unusual-looking legitimate links and well-crafted phishing links. Treat the result "
    "as a screening signal, not a guarantee."
)
