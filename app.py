"""
Phishing URL Detector
=====================
Loads the trained Random Forest (phishing_random_forest_model.pkl) and lets a user
paste a URL: the app auto-fills the model's 81 features (see features.py), shows
where each value came from, lets the user edit any of them, then classifies the page
as legitimate or phishing.

Files needed next to this script:
    phishing_random_forest_model.pkl   the model saved by the notebook (rf_model)
    dataset_B_05_2020.csv              used for the Overview / Performance pages and for
                                       neutral fallback values (training medians)
    features.py                        URL -> feature extraction
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

import features as F

RANDOM_STATE = 42
HERE = Path(__file__).parent
DATA_PATH = HERE / "dataset_B_05_2020.csv"
MODEL_PATH = HERE / "phishing_random_forest_model.pkl"
LABELS = {0: "legitimate", 1: "phishing"}

st.set_page_config(page_title="Phishing URL Detector", page_icon="🎣", layout="wide")

BADGE = {"url": "✅", "page": "🌐", "lookup": "🔎", "estimated": "⚠️", "data": "📚"}
BADGE_TEXT = {
    "url": "computed from the URL",
    "page": "read from the page's HTML",
    "lookup": "from a DNS / RDAP / PageRank lookup",
    "estimated": "estimated (training-set median) - please check",
    "data": "true value from the dataset",
}
HELP = {
    "google_index": "1 = the page is NOT indexed by Google, 0 = indexed. Strongest feature in the model.",
    "page_rank": "Open PageRank score, 0-10. Low values are typical of phishing pages.",
    "web_traffic": "Traffic rank of the domain (0 = unranked). Open PageRank's global rank is used as a stand-in.",
    "domain_age": "Days since the domain was registered.",
    "domain_registration_length": "Days until the domain registration expires.",
    "dns_record": "1 = the domain has NO DNS record.",
    "whois_registered_domain": "1 = no WHOIS/RDAP record was found.",
    "nb_hyperlinks": "Links, scripts, images, forms and stylesheets found in the page.",
    "statistical_report": "PhishTank-based statistics. Not reproducible offline, so it's left at the training median.",
    "random_domain": "Whether the domain looks randomly generated. Left at the training median.",
}


# ----------------------------------------------------------------------------
# Data + model (cached so the app doesn't reload on every click)
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Training the tuned model (a few seconds, once)...")
def load_tuned():
    """The notebook's GridSearchCV winner (n_estimators=200, max_features='log2', random_state=42),
    refit on the same 80% training split. Deterministic, so it reproduces the notebook's results.
    The notebook only saved the initial model; to ship the tuned one as a file instead, add
    joblib.dump(rf_model_tuned, 'phishing_random_forest_tuned.pkl') to the notebook."""
    from sklearn.ensemble import RandomForestClassifier
    d = load_data()
    return RandomForestClassifier(n_estimators=200, max_features="log2", random_state=RANDOM_STATE).fit(
        d["X_train"], d["y_train"])


MODEL_LABELS = {"tuned": "Tuned Random Forest (200 trees)", "initial": "Initial Random Forest (100 trees, saved .pkl)"}


def get_model(which):
    return load_tuned() if which == "tuned" else load_model()


@st.cache_data(show_spinner="Preparing data...")
def load_data():
    """Same preprocessing as the notebook: drop duplicate URLs, encode the target,
    stratified 80/20 split with random_state=42, keep the model's 81 columns."""
    raw = pd.read_csv(DATA_PATH).drop_duplicates(subset=["url"]).reset_index(drop=True)
    raw["target"] = raw["status"].map({"legitimate": 0, "phishing": 1})
    cols = list(load_model().feature_names_in_)
    X, y = raw[cols], raw["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    urls_test = raw.loc[X_test.index, "url"]
    return {"raw": raw, "cols": cols, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "urls_test": urls_test}


@st.cache_data(show_spinner="Evaluating model...")
def evaluate(which):
    model, d = get_model(which), load_data()
    y_pred = model.predict(d["X_test"])
    y_proba = model.predict_proba(d["X_test"])[:, 1]
    yt = d["y_test"]
    return {
        "y_pred": y_pred, "y_proba": y_proba,
        "metrics": {
            "accuracy": accuracy_score(yt, y_pred), "precision": precision_score(yt, y_pred),
            "recall": recall_score(yt, y_pred), "f1": f1_score(yt, y_pred),
            "auc": roc_auc_score(yt, y_proba),
        },
    }


@st.cache_data(show_spinner="Measuring auto-fill reliability...")
def fallback_accuracies(which):
    """Test-set accuracy under the two ways the auto-fill can fall short:
      url_only        only the URL-derived features are real; every page / look-up feature
                      is replaced by its training median (page offline, no look-ups at all)
      reputation_gap  everything is measured except google_index, page_rank and web_traffic,
                      which need a PageRank key and are filled with their training medians"""
    model, d = get_model(which), load_data()
    medians = d["X_train"].median()
    y = d["y_test"]

    X = d["X_test"].copy()
    lex = pd.DataFrame([F.lexical_features(u) for u in d["urls_test"]], index=X.index)
    for c in X.columns:
        X[c] = lex[c] if c in lex.columns else medians[c]
    url_only = accuracy_score(y, model.predict(X))

    X2 = d["X_test"].copy()
    for c in ("google_index", "page_rank", "web_traffic"):
        X2[c] = medians[c]
    return url_only, accuracy_score(y, model.predict(X2))


data = load_data()
COLS = data["cols"]
MEDIANS = data["X_train"].median().to_dict()


def kind_of(col: str) -> str:
    s = data["raw"][col]
    if set(s.unique()) <= {0, 1}:
        return "bin"
    return "int" if np.allclose(s, s.round()) else "float"


KINDS = {c: kind_of(c) for c in COLS}


def cast(col, v):
    k = KINDS[col]
    if k == "bin":
        return int(round(min(max(float(v), 0), 1)))
    return int(round(float(v))) if k == "int" else float(v)


def get_opr_key() -> str:
    try:
        return st.secrets.get("OPENPAGERANK_API_KEY", "")
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("Phishing URL Detector")
st.sidebar.markdown(
    "Classifies a web page as **legitimate** or **phishing** with a Random Forest "
    "trained on URL, page-content and domain features."
)
which = st.sidebar.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get,
                             help="Used for predictions. The Performance page always compares both.")
model = get_model(which)
IMPORTANCE = pd.Series(model.feature_importances_, index=COLS)
st.sidebar.markdown(f"**Training rows:** {len(data['X_train']):,}")
st.sidebar.markdown(f"**Features:** {len(COLS)}")
st.sidebar.caption("A scikit-learn Random Forest, deployed with Streamlit.")
page = st.sidebar.radio("Go to", ["Overview", "Try a Prediction", "Model Performance"])

opr_key = get_opr_key()
if page == "Try a Prediction":
    st.sidebar.divider()
    opr_key = st.sidebar.text_input(
        "Open PageRank API key (optional)", value=opr_key, type="password",
        help="Free key from openpagerank.com. Enables the page_rank, web_traffic and "
             "google_index look-ups, which are the model's most important features.",
    )
    do_fetch = st.sidebar.toggle(
        "Fetch the page & run look-ups", value=True,
        help="Turn off to compute only the URL-based features. The app never visits the page's links.",
    )


# ----------------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------------

if page == "Overview":
    raw = data["raw"]
    st.title("🎣 Phishing URL Detector")
    st.markdown(
        "This app uses a Random Forest trained on **11,429 labelled web pages** to flag "
        "each one as **legitimate** or **phishing**. Paste a URL on the *Try a Prediction* "
        "page and the app fills in the model's features for you."
    )

    acc = evaluate(which)["metrics"]["accuracy"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Phishing rate", f"{raw['target'].mean():.1%}")
    c3.metric("Features used", len(COLS))
    c4.metric("Test accuracy", f"{acc:.1%}")

    st.subheader("Class balance")
    st.bar_chart(raw["status"].value_counts())

    st.subheader("Sample of the training data")
    st.dataframe(raw.drop(columns=["target"]).head(20), width="stretch")

    st.subheader("Where the model's signal actually comes from")
    top = IMPORTANCE.sort_values(ascending=False).head(5)
    gi = raw.groupby("status")["google_index"].mean()
    pr = raw.groupby("status")["page_rank"].mean()
    st.info(
        "The most influential features are not about the URL text at all: "
        + ", ".join(f"**`{c}`** ({v:.0%})" for c, v in top.items())
        + f" of the model's importance. Pages that Google hasn't indexed are far more often phishing "
        f"({gi['phishing']:.0%} of phishing pages vs {gi['legitimate']:.0%} of legitimate ones), and the average "
        f"PageRank is {pr['legitimate']:.1f} for legitimate pages vs {pr['phishing']:.1f} for phishing. "
        "That's why the auto-fill makes live look-ups instead of reading the URL alone - see **Model Performance** "
        "for how much accuracy depends on them."
    )

    st.subheader("How the auto-fill works")
    st.markdown(
        "| Icon | Source | Examples |\n|---|---|---|\n"
        "| ✅ | Computed from the URL string | length, dots, hyphens, digits, suspicious words |\n"
        "| 🌐 | Read from the page's HTML | number of links, login form, title, iframes |\n"
        "| 🔎 | Looked up online | DNS record, domain age (RDAP), PageRank |\n"
        "| ⚠️ | Couldn't be obtained | filled with the training median - edit it if you know better |"
    )


# ----------------------------------------------------------------------------
# Page: Try a Prediction
# ----------------------------------------------------------------------------

elif page == "Try a Prediction":
    st.title("Try a Prediction")
    st.markdown("Paste a URL to auto-fill the features, or upload a CSV to classify many at once.")

    tab1, tab2 = st.tabs(["Check a URL", "Upload CSV"])

    # Every widget default lives in session_state (a widget with a `key` can't also take `value=`).
    for c in COLS:
        st.session_state.setdefault(f"f_{c}", cast(c, MEDIANS[c]))
    st.session_state.setdefault("url_input", "")
    st.session_state.setdefault("prov", {c: "estimated" for c in COLS})
    st.session_state.setdefault("notes", [])
    st.session_state.setdefault("analysed_url", "")

    def load_random_example():
        label = int(np.random.choice([0, 1]))
        row = data["X_train"][data["y_train"] == label].sample(1).iloc[0]
        for c in COLS:
            st.session_state[f"f_{c}"] = cast(c, row[c])
        st.session_state["prov"] = {c: "data" for c in COLS}
        st.session_state["notes"] = [f"Loaded a real {LABELS[label]} example from the dataset (true values, no look-ups)."]
        st.session_state["analysed_url"] = ""

    with tab1:
        col_u, col_b = st.columns([5, 1], vertical_alignment="bottom")
        with col_u:
            st.text_input(
                "URL to check", key="url_input", placeholder="https://example.com/login",
                help="Never open links you suspect are phishing. This app only requests the page's HTML.",
            )
        with col_b:
            analyse = st.button("🔍 Analyze", type="primary", width="stretch")

        st.button("🎲 Load a random example from the dataset", on_click=load_random_example)

        if analyse:
            if not st.session_state["url_input"].strip():
                st.warning("Enter a URL first.")
            else:
                with st.spinner("Extracting features (page fetch, DNS, RDAP)..."):
                    feats, prov, notes = F.extract_all(
                        st.session_state["url_input"], MEDIANS, opr_key=opr_key, fetch=do_fetch
                    )
                for c in COLS:
                    st.session_state[f"f_{c}"] = cast(c, feats[c])
                st.session_state["prov"], st.session_state["notes"] = prov, notes
                st.session_state["analysed_url"] = F.normalise_url(st.session_state["url_input"])

        prov = st.session_state["prov"]
        if st.session_state["analysed_url"]:
            counts = pd.Series(prov).value_counts()
            st.markdown(
                f"**Auto-filled** `{st.session_state['analysed_url']}` - "
                + " · ".join(f"{BADGE[k]} {counts.get(k, 0)} {k}" for k in BADGE)
            )
        for n in st.session_state["notes"]:
            st.caption(f"ℹ️ {n}")

        st.markdown("Review or edit any value below. " + "  ".join(f"{BADGE[k]} {v}" for k, v in BADGE_TEXT.items()))

        groups = {
            "🔗 URL structure": [c for c in COLS if c not in F.CONTENT_COLS + F.EXTERNAL_COLS + F.ALWAYS_ESTIMATED],
            "📄 Page content": [c for c in COLS if c in F.CONTENT_COLS],
            "🌍 Domain & reputation": [c for c in COLS if c in F.EXTERNAL_COLS + F.ALWAYS_ESTIMATED],
        }
        for title, cols in groups.items():
            n_est = sum(prov.get(c) == "estimated" for c in cols)
            with st.expander(f"{title} ({len(cols)})" + (f" - {n_est} estimated" if n_est else ""),
                             expanded=title.startswith("🌍")):
                grid = st.columns(3)
                for i, c in enumerate(sorted(cols, key=lambda x: -IMPORTANCE[x])):
                    label = f"{BADGE[prov.get(c, 'estimated')]} {c}"
                    with grid[i % 3]:
                        k = KINDS[c]
                        if k == "bin":
                            st.number_input(label, min_value=0, max_value=1, step=1, key=f"f_{c}", help=HELP.get(c))
                        elif k == "int":
                            st.number_input(label, step=1, key=f"f_{c}", help=HELP.get(c))
                        else:
                            st.number_input(label, step=0.01, format="%.3f", key=f"f_{c}", help=HELP.get(c))

        if st.button("Classify this page", type="primary"):
            record = pd.DataFrame([{c: st.session_state[f"f_{c}"] for c in COLS}])[COLS]
            pred = int(model.predict(record)[0])
            proba = float(model.predict_proba(record)[0, 1])

            st.subheader("Prediction: " + ("🚨 Phishing" if pred else "✅ Legitimate"))
            st.metric("Predicted phishing probability", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))

            est_share = sum(IMPORTANCE[c] for c in COLS if prov.get(c) == "estimated")
            if est_share > 0.25:
                st.warning(
                    f"{est_share:.0%} of the model's weight sits on features that were **estimated**, not measured "
                    "(most likely google_index / page_rank / web_traffic). Treat this result as indicative - "
                    "add an Open PageRank key in the sidebar or edit those fields manually."
                )

            f = {c: st.session_state[f"f_{c}"] for c in COLS}
            signals = []
            if f["google_index"] == 1: signals.append("the page is not indexed by Google")
            if f["page_rank"] <= 1: signals.append("very low PageRank")
            if f["web_traffic"] == 0: signals.append("no web-traffic ranking")
            if 0 <= f["domain_age"] < 365: signals.append("the domain is under a year old")
            if f["ip"] == 1: signals.append("the URL contains an IP address / long hex string")
            if f["nb_hyperlinks"] < 10: signals.append("the page has very few links")
            if f["phish_hints"] > 0: signals.append("the URL path contains words like login/signin/admin")
            if f["suspecious_tld"] == 1: signals.append("the TLD is commonly abused")
            if f["shortening_service"] == 1: signals.append("a URL-shortening service is used")
            if f["login_form"] == 1: signals.append("a form posts to an empty or external target")
            if signals:
                st.caption("Signals present: " + "; ".join(signals) + ".")
            st.caption("This is a statistical model, not a verdict. Use it alongside other checks.")

    with tab2:
        st.markdown(
            "Upload a CSV with **either** the 81 feature columns **or** a `url` column. "
            "With `url` only, features are extracted automatically (max 25 rows per upload, since each row "
            "needs live look-ups)."
        )
        uploaded = st.file_uploader("Upload records (CSV)", type="csv")
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            if set(COLS) <= set(new_df.columns):
                X_new = new_df[COLS]
            elif "url" in new_df.columns:
                new_df = new_df.dropna(subset=["url"]).reset_index(drop=True)
                urls = new_df["url"].astype(str).head(25).tolist()
                if len(new_df) > 25:
                    st.info("Only the first 25 URLs were processed.")
                rows, bar = [], st.progress(0.0, text="Extracting features...")
                for i, u in enumerate(urls):
                    rows.append(F.extract_all(u, MEDIANS, opr_key=opr_key, fetch=do_fetch)[0])
                    bar.progress((i + 1) / len(urls), text=f"Extracting features... {i + 1}/{len(urls)}")
                bar.empty()
                new_df = new_df.head(25).reset_index(drop=True)
                X_new = pd.DataFrame(rows)[COLS]
            else:
                X_new = None
                st.error("The CSV needs a `url` column or all of the model's feature columns.")

            if X_new is not None:
                out = new_df[["url"]].copy() if "url" in new_df.columns else pd.DataFrame(index=new_df.index)
                out["phishing_probability"] = model.predict_proba(X_new)[:, 1]
                out["prediction"] = pd.Series(model.predict(X_new), index=out.index).map(LABELS)
                n_ph = int((out["prediction"] == "phishing").sum())
                st.success(f"Classified {len(out)} rows -- {n_ph} flagged as phishing.")
                st.dataframe(
                    out.style.map(lambda v: "background-color: #ffcccc" if v == "phishing" else "", subset=["prediction"]),
                    width="stretch",
                )
                st.download_button("Download results as CSV", out.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")


# ----------------------------------------------------------------------------
# Page: Model Performance
# ----------------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")
    evs = {k: evaluate(k) for k in MODEL_LABELS}
    yt = data["y_test"]
    st.markdown(
        f"Evaluated on a held-out test set of **{len(yt):,}** pages (never seen during training). "
        "Both Random Forests are shown: the **initial** model saved by the notebook and the **tuned** model "
        "(GridSearchCV: 200 trees, `max_features='log2'`)."
    )

    rows = []
    for k, label in MODEL_LABELS.items():
        m = evs[k]["metrics"]
        rows.append({"Model": label, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "AUC": m["auc"]})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.4f}"), width="stretch")

    m = evs[which]["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1 Score", f"{m['f1']:.3f}")
    c5.metric("AUC", f"{m['auc']:.3f}")
    st.caption(f"Headline metrics for the model selected in the sidebar: {MODEL_LABELS[which]}.")

    url_only, rep_gap = fallback_accuracies(which)
    st.info(
        "These scores use the dataset's **true** feature values. In the app, the URL-based features are rebuilt "
        "from the URL and the rest come from live look-ups, so real-world accuracy depends on how many succeed. "
        f"On the same test set, if only the URL is usable, accuracy falls to **{url_only:.1%}**; if everything is "
        f"measured except `google_index`, `page_rank` and `web_traffic` (no PageRank key), it is **{rep_gap:.1%}**. "
        "Add an Open PageRank key to close most of that gap."
    )

    st.subheader("Confusion Matrix")
    cols_cm = st.columns(2)
    for col, (k, label) in zip(cols_cm, MODEL_LABELS.items()):
        with col:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4.4))
            ConfusionMatrixDisplay(confusion_matrix(yt, evs[k]["y_pred"]),
                                   display_labels=["Legitimate", "Phishing"]).plot(ax=ax_cm)
            ax_cm.set_title(label.split(" (")[0] + " Confusion Matrix")
            st.pyplot(fig_cm)

    st.subheader("ROC Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 5.5))
    styles = {"initial": ("darkorange", "-"), "tuned": ("navy", "--")}
    for k in MODEL_LABELS:
        fpr, tpr, _ = roc_curve(yt, evs[k]["y_proba"])
        color, ls = styles[k]
        ax_roc.plot(fpr, tpr, color=color, linestyle=ls, lw=2,
                    label=f"{k.capitalize()} RF (AUC = {evs[k]['metrics']['auc']:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.05)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax_roc.grid(True, linestyle="--", alpha=0.5); ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    st.subheader("Top 20 Most Important Features")
    imp = pd.Series(model.feature_importances_, index=COLS).sort_values(ascending=False).head(20)
    fig_imp, ax_imp = plt.subplots(figsize=(9, 7))
    ax_imp.barh(imp.index[::-1], imp.values[::-1])
    ax_imp.set_xlabel("Feature Importance"); ax_imp.set_ylabel("Feature")
    st.pyplot(fig_imp)
    st.caption(f"Importances from: {MODEL_LABELS[which]}.")

    st.subheader("Sample test-set predictions")
    idx = data["X_test"].index[:10]
    sample = pd.DataFrame({
        "url": data["urls_test"].loc[idx].str.slice(0, 70),
        "true_label": yt.loc[idx].map(LABELS),
        "predicted_label": pd.Series(evs[which]["y_pred"], index=data["X_test"].index).loc[idx].map(LABELS),
    })
    st.dataframe(sample, width="stretch")
