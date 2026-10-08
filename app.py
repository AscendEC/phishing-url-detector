import joblib
import pandas as pd
import streamlit as st

from features import extract_url_features

MODEL_PATH = "phishing_random_forest_model.pkl"

st.set_page_config(page_title="Phishing Detector", page_icon="🎣", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
FEATURES = list(model.feature_names_in_)

# Feature groups, used to organise the input form
URL_FEATURES = set(extract_url_features("http://example.com").keys())
EXTERNAL = {"whois_registered_domain", "domain_registration_length", "domain_age",
            "web_traffic", "dns_record", "google_index", "page_rank",
            "statistical_report"}
CONTENT = [f for f in FEATURES if f not in URL_FEATURES and f not in EXTERNAL]
GROUPS = {
    "🔗 URL-based features": [f for f in FEATURES if f in URL_FEATURES],
    "📄 Page-content features": [f for f in FEATURES if f in CONTENT],
    "🌐 External / domain features": [f for f in FEATURES if f in EXTERNAL],
}

# Sidebar
st.sidebar.header("Settings")
phish_label = st.sidebar.selectbox(
    "Class label meaning 'phishing'", list(model.classes_),
    index=len(model.classes_) - 1,
    help="Usually 1 = phishing, 0 = legitimate. Change if your encoding is reversed.",
)
threshold = st.sidebar.slider("Phishing threshold", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption(f"Random Forest · {len(model.estimators_)} trees · {len(FEATURES)} features")


def phish_probability(df: pd.DataFrame):
    idx = list(model.classes_).index(phish_label)
    return model.predict_proba(df[FEATURES])[:, idx]


# Session state defaults
for f in FEATURES:
    st.session_state.setdefault(f"feat_{f}", 0.0)


def fill_from_url():
    url = st.session_state.get("url_input", "")
    if not url.strip():
        return
    for k, v in extract_url_features(url).items():
        if k in FEATURES:
            st.session_state[f"feat_{k}"] = v


def reset_inputs():
    for f in FEATURES:
        st.session_state[f"feat_{f}"] = 0.0


st.title("🎣 Phishing Website Detector")
st.write("Random Forest classifier trained on 81 URL, page-content and domain features.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch CSV"])

# ---------------- Single prediction ----------------
with tab_single:
    st.subheader("1. (Optional) Pre-fill URL features from a link")
    c1, c2, c3 = st.columns([5, 1, 1])
    c1.text_input("URL", key="url_input", placeholder="https://example.com/login")
    c2.button("Extract", on_click=fill_from_url, use_container_width=True)
    c3.button("Reset", on_click=reset_inputs, use_container_width=True)
    st.info(
        "Only URL-based features can be computed from the link. Page-content and "
        "external features (e.g. `google_index`, `page_rank`, `web_traffic`, "
        "`domain_age`) are among the model's most important inputs, so fill them in "
        "below for a trustworthy result."
    )

    st.subheader("2. Review / edit features")
    for title, feats in GROUPS.items():
        with st.expander(title, expanded=title.startswith("🌐")):
            cols = st.columns(3)
            for i, f in enumerate(feats):
                cols[i % 3].number_input(f, key=f"feat_{f}", format="%.4f")

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{f: st.session_state[f"feat_{f}"] for f in FEATURES}])
        p = float(phish_probability(row)[0])
        if p >= threshold:
            st.error(f"⚠️ Likely PHISHING — probability {p:.1%}")
        else:
            st.success(f"✅ Likely LEGITIMATE — phishing probability {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

# ---------------- Batch ----------------
with tab_batch:
    st.write(
        "Upload a CSV containing the model's feature columns. Extra columns "
        "(e.g. `url`, `status`) are kept in the output and ignored by the model."
    )
    template = pd.DataFrame(columns=FEATURES).to_csv(index=False).encode()
    st.download_button("Download CSV template", template, "template.csv", "text/csv")

    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"Missing {len(missing)} required column(s): {', '.join(missing[:10])}"
                     + (" ..." if len(missing) > 10 else ""))
        else:
            probs = phish_probability(df)
            out = df.copy()
            out["phishing_probability"] = probs
            out["prediction"] = ["phishing" if p >= threshold else "legitimate" for p in probs]
            st.metric("Flagged as phishing", f"{(probs >= threshold).sum()} / {len(out)}")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")
