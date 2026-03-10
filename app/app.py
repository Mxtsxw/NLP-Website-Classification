"""
NLP Web Page Classifier — Streamlit Dashboard
All artefacts (models, feature matrices, data) are loaded from S3.
No local data files or model training at runtime.
"""

import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

from s3_client import load_all_artefacts

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
CLASSES  = ["FAQ", "accueil", "blog", "commerce", "home", "liste", "recherche"]
PALETTE  = ["#7c6aff", "#ff6a9e", "#6affd4", "#ffd06a", "#ff9d6a", "#6ab8ff", "#d46aff"]
STRUCT_COLS = ["n_links", "n_forms", "n_inputs", "n_tables",
               "n_imgs", "n_headings", "n_list_items", "text_len"]

PLOTLY_THEME = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#f7f8fa",
    font=dict(family="Inter, sans-serif", color="#1a1a2e", size=11),
    xaxis=dict(gridcolor="#e4e6ec", linecolor="#e4e6ec", tickfont=dict(color="#888aa0")),
    yaxis=dict(gridcolor="#e4e6ec", linecolor="#e4e6ec", tickfont=dict(color="#888aa0")),
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NLP Web Page Classifier", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #f7f8fa;
      color: #1a1a2e;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
      gap: 0;
      background: #ffffff;
      border: 1px solid #e4e6ec;
      border-radius: 8px;
      padding: 3px;
  }
  .stTabs [data-baseweb="tab"] {
      padding: .55rem 1.3rem;
      color: #888aa0;
      font-size: .75rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      border-radius: 6px;
      border: none;
      background: transparent;
      font-family: 'Inter', sans-serif;
      font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
      background: #1a1a2e;
      color: #ffffff;
  }

  /* ── Metric cards ── */
  .metric-card {
      background: #ffffff;
      border: 1px solid #e4e6ec;
      border-left: 3px solid;
      padding: 1.2rem 1.3rem;
      border-radius: 10px;
  }
  .metric-card.purple { border-left-color: #5b4fcf; }
  .metric-card.pink   { border-left-color: #c0392b; }
  .metric-card.teal   { border-left-color: #1e7a4a; }
  .metric-card.gold   { border-left-color: #b7700a; }
  .metric-label {
      font-size: .62rem;
      color: #888aa0;
      text-transform: uppercase;
      letter-spacing: .12em;
      margin-bottom: .35rem;
      font-weight: 500;
  }
  .metric-value {
      font-size: 1.9rem;
      font-weight: 700;
      color: #1a1a2e;
      letter-spacing: -.02em;
      line-height: 1.1;
  }
  .metric-sub {
      font-family: 'JetBrains Mono', monospace;
      font-size: .65rem;
      color: #aaaacc;
      margin-top: .3rem;
  }

  /* ── Headings ── */
  h1 {
      font-family: 'Inter', sans-serif !important;
      font-size: 1.9rem !important;
      font-weight: 700 !important;
      letter-spacing: -.02em !important;
      color: #1a1a2e !important;
  }
  h2 {
      font-family: 'Inter', sans-serif !important;
      font-size: 1rem !important;
      font-weight: 600 !important;
      color: #1a1a2e !important;
  }
  h3 {
      font-family: 'Inter', sans-serif !important;
      font-size: .62rem !important;
      font-weight: 500 !important;
      color: #aaaacc !important;
      text-transform: uppercase !important;
      letter-spacing: .18em !important;
      border-bottom: 1px solid #e4e6ec;
      padding-bottom: 6px;
      margin: 26px 0 14px !important;
  }

  /* ── Sidebar ── */
  .stSidebar {
      background: #ffffff;
      border-right: 1px solid #e4e6ec;
  }

  /* ── Buttons ── */
  .stButton > button {
      background: #1a1a2e;
      color: #ffffff;
      border: none;
      border-radius: 6px;
      font-family: 'Inter', sans-serif;
      font-size: .75rem;
      font-weight: 500;
      letter-spacing: .06em;
      text-transform: uppercase;
      padding: .5rem 1.2rem;
  }
  .stButton > button:hover { background: #2c2c4a; }

  /* ── Native st.metric override ── */
  [data-testid="metric-container"] {
      background: #ffffff;
      border: 1px solid #e4e6ec;
      border-radius: 10px;
      padding: 16px;
  }
  [data-testid="metric-container"] label {
      color: #888aa0 !important;
      font-size: .7rem !important;
      letter-spacing: .08em;
      text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-size: 1.7rem !important;
      font-weight: 700;
      color: #1a1a2e;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
      font-family: 'JetBrains Mono', monospace;
      font-size: .72rem;
  }

  /* ── Section headers ── */
  .section-header {
      font-size: .62rem;
      letter-spacing: .18em;
      text-transform: uppercase;
      color: #aaaacc;
      border-bottom: 1px solid #e4e6ec;
      padding-bottom: 6px;
      margin: 26px 0 14px;
  }

  /* ── Insight cards ── */
  .insight-card {
      background: #ffffff;
      border: 1px solid #e4e6ec;
      border-left: 3px solid #c0392b;
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 10px;
      font-size: .84rem;
      line-height: 1.65;
      color: #444466;
  }
  .insight-card .tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: .63rem;
      background: #fdecea;
      color: #c0392b;
      padding: 2px 8px;
      border-radius: 4px;
      letter-spacing: .08em;
      margin-bottom: 6px;
      display: inline-block;
  }
  .insight-card.blue  { border-left-color: #2c5f8a; }
  .insight-card.blue  .tag { background: #eaf0f7; color: #2c5f8a; }
  .insight-card.amber { border-left-color: #b7700a; }
  .insight-card.amber .tag { background: #fdf3e3; color: #b7700a; }
  .insight-card.green { border-left-color: #1e7a4a; }
  .insight-card.green .tag { background: #eaf5f0; color: #1e7a4a; }

  /* ── S3 badge ── */
  .s3-badge {
      display: inline-flex;
      align-items: center;
      gap: .4rem;
      background: #eaf5f0;
      border: 1px solid #b8deca;
      border-radius: 4px;
      padding: .2rem .65rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: .63rem;
      color: #1e7a4a;
      letter-spacing: .06em;
  }

  /* ── Status dot ── */
  .status-dot {
      display: inline-block;
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #1e7a4a;
      margin-right: 6px;
  }

  /* ── Dashboard title helpers ── */
  .dash-title {
      font-size: 1.9rem;
      font-weight: 700;
      letter-spacing: -.02em;
      color: #1a1a2e;
  }
  .dash-subtitle {
      color: #aaaacc;
      font-size: .75rem;
      letter-spacing: .12em;
      text-transform: uppercase;
  }

  /* ── Chrome cleanup ── */
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─── Load all artefacts from S3 (cached — runs once per session) ──────────────
with st.spinner("Pulling artefacts from S3…"):
    art = load_all_artefacts()

tfidf         = art["tfidf"]
svd           = art["svd"]
le            = art["le"]
knn_model     = art["knn"]
svm_model     = art["svm"]
X_train_knn   = art["X_train_knn"]
X_test_knn    = art["X_test_knn"]
X_train_tfidf = art["X_train_tfidf"]
X_test_tfidf  = art["X_test_tfidf"]
y_train       = art["y_train"]
y_test        = art["y_test"]
train_df      = art["train_features"].merge(
                    art["train_texts"][["file", "text"]], on="file", how="left")
test_df       = art["test_features"].merge(
                    art["test_texts"][["file", "text"]], on="file", how="left")
results       = art["model_results"]

vocab      = tfidf.get_feature_names_out()
knn_acc    = results["knn_acc"]
svm_acc    = results["svm_acc"]
knn_report = results["knn_report"]
svm_report = results["svm_report"]
knn_cm     = np.array(results["knn_cm"])
svm_cm     = np.array(results["svm_cm"])


# ─── Derived computations ───
@st.cache_data(show_spinner=False)
def _svd2d(_X):
    return TruncatedSVD(n_components=2, random_state=42).fit_transform(_X)

@st.cache_data(show_spinner=False)
def _word_counts(_texts, _labels):
    cv = CountVectorizer(max_features=5000, stop_words="english", min_df=2)
    cv.fit(_texts)
    return cv.get_feature_names_out(), {
        cls: cv.transform(_texts[_labels == cls]).toarray().sum(axis=0)
        for cls in CLASSES
    }

X2d = _svd2d(X_train_tfidf)
cv_vocab, word_counts = _word_counts(
    train_df["text"].fillna("").values, train_df["label"].values
)


# ─── Sidebar ───
with st.sidebar:
    st.markdown("## Web Page Classifier")
    st.markdown('<div class="s3-badge">artefacts from S3</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
**Dataset**
- 1050 HTML documents · 7 classes
- 840 train / 210 test

**Models**
- KNN (k=5, cosine, LSA+struct)
- LinearSVC (TF-IDF bigrams)
""")
    st.markdown("---")
    st.markdown(f"**KNN** `{knn_acc*100:.1f}%`"); st.progress(knn_acc)
    st.markdown(f"**SVM** `{svm_acc*100:.1f}%`"); st.progress(svm_acc)
    st.markdown("---")
    st.markdown("### Classify New Text")
    user_text = st.text_area("Paste HTML or plain text:", height=120,
                              placeholder="<html>...</html> or plain text")
    if st.button("Classify") and user_text.strip():
        soup = BeautifulSoup(user_text)
        for t in soup(["script", "style"]): t.decompose()
        clean = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()
        pred  = svm_model.predict(tfidf.transform([clean]))[0]
        label = le.inverse_transform([pred])[0]
        ci    = CLASSES.index(label)
        st.markdown(f"""
        <div style="background:#f5f5ff;border:1px solid {PALETTE[ci]};border-radius:4px;padding:1rem;margin-top:.5rem;">
          <div style="font-size:.65rem;color:#9090b0;text-transform:uppercase;letter-spacing:.1em;">Predicted class</div>
          <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:{PALETTE[ci]}">{label}</div>
          <div style="font-size:.65rem;color:#9090b0;margin-top:.3rem;">via SVM - LinearSVC</div>
        </div>""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# Web Page NLP Classifier")
st.markdown('End-to-end text classification · EDA · Feature engineering · KNN & SVM &nbsp;<span class="s3-badge">artefacts from S3</span>', unsafe_allow_html=True)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
for col, cls, label, val, sub in [
    (c1, "purple", "Total Documents", "1050",               "840 train / 210 test"),
    (c2, "pink",   "Classes",         "7",                  "Perfectly balanced"),
    (c3, "teal",   "SVM Accuracy",    f"{svm_acc*100:.1f}%","LinearSVC + TF-IDF"),
    (c4, "gold",   "KNN Accuracy",    f"{knn_acc*100:.1f}%","LSA + cosine"),
]:
    col.markdown(f"""<div class="metric-card {cls}">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{val}</div>
  <div class="metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

st.markdown("")
tab_eda, tab_feat, tab_models, tab_words = st.tabs(
    ["EDA", "Feature Engineering", "Models", "Top Words"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("### Class Distribution")
    counts = [int((train_df["label"] == c).sum()) for c in CLASSES]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=CLASSES, y=counts, marker_color=PALETTE))
        fig.update_layout(title="Training Set Class Distribution", **PLOTLY_THEME,
                          height=320, margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Pie(labels=CLASSES, values=counts,
                               marker=dict(colors=PALETTE, line=dict(color="#0a0a0f", width=2)),
                               hole=0.5, textfont_size=10)
                        )
        fig.update_layout(title="Class Proportions", height=320,
                          margin=dict(l=20, r=20, t=50, b=20), **PLOTLY_THEME,
                          legend=dict(font=dict(size=10))
                          )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Text Length Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for cls, color in zip(CLASSES, PALETTE):
            fig.add_trace(go.Box(y=train_df[train_df["label"]==cls]["text_len"].dropna().values,
                                 name=cls, marker_color=color, line_color=color,
                                 fillcolor=color))
        fig.update_layout(title="Text Length by Class (chars)", yaxis_title="Characters",
                          **PLOTLY_THEME, height=360, showlegend=False,
                          margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        mean_words = [train_df[train_df["label"]==c]["n_words"].mean() for c in CLASSES]
        fig = go.Figure(go.Bar(x=mean_words, y=CLASSES, orientation="h",
                               marker_color=PALETTE, marker_line_color="#0a0a0f",
                               text=[f"{v:.0f}" for v in mean_words], textposition="outside"))
        fig.update_layout(title="Mean Word Count per Category", xaxis_title="Average Words",
                          **PLOTLY_THEME, height=360, margin=dict(l=80, r=60, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SVD 2D Projection — Class Separability")
    fig = go.Figure()
    for cls, color in zip(CLASSES, PALETTE):
        mask = train_df["label"].values == cls
        fig.add_trace(go.Scatter(x=X2d[mask, 0], y=X2d[mask, 1], mode="markers", name=cls,
                                 marker=dict(color=color, size=5, opacity=0.6, line=dict(width=0))))
    fig.update_layout(title="TF-IDF Projected to 2D via Truncated SVD",
                      xaxis_title="SVD Component 1", yaxis_title="SVD Component 2",
                      **PLOTLY_THEME, height=450, margin=dict(l=40, r=20, t=50, b=40),
                      legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Structural Feature Correlation Heatmap")
    num_cols = ["text_len","n_words","n_links","n_forms","n_inputs",
                "n_tables","n_imgs","n_headings","n_list_items"]
    corr = train_df[num_cols].corr().round(2)
    fig  = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(),
                                y=corr.columns.tolist(), colorscale="RdBu", zmid=0,
                                text=corr.values.round(2), texttemplate="%{text}",
                                textfont=dict(size=9), colorbar=dict(len=0.8)))
    fig.update_layout(title="Feature Correlation Matrix", **PLOTLY_THEME, height=420,
                      margin=dict(l=100, r=40, t=50, b=80))
    fig.update_xaxes(tickangle=40, tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════════
with tab_feat:
    st.markdown("### TF-IDF Vectorization")
    col1, col2, col3 = st.columns(3)
    col1.metric("Vocabulary Size", "10,000", "max_features")
    col2.metric("N-gram Range", "1-2", "unigrams + bigrams")
    col3.metric("Sublinear TF", "Enabled", "log(1 + tf)")
    n_uni    = sum(1 for w in vocab if " " not in w)
    sparsity = 1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])
    st.info(f"Matrix: {X_train_tfidf.shape}  |  Sparsity: {sparsity:.1%}  |  Unigrams: {n_uni}  |  Bigrams: {len(vocab)-n_uni}")

    st.markdown("### Mean TF-IDF Score — Top Terms per Class")
    fig = make_subplots(rows=2, cols=4, subplot_titles=CLASSES,
                        horizontal_spacing=0.06, vertical_spacing=0.14)
    for i, (cls, color) in enumerate(zip(CLASSES, PALETTE)):
        row, col = divmod(i, 4)
        idx = np.where(le.transform(train_df["label"]) == le.transform([cls])[0])[0]
        ct  = X_train_tfidf[idx].mean(axis=0).A1
        top = ct.argsort()[::-1][:10]
        fig.add_trace(go.Bar(x=ct[top][::-1].tolist(), y=[vocab[j] for j in top][::-1],
                             orientation="h", name=cls, showlegend=False),
                      row=row+1, col=col+1)
    fig.update_layout(**PLOTLY_THEME, height=560, margin=dict(l=10,r=10,t=60,b=20),
                      title="Top Terms by Mean TF-IDF Weight per Class")
    fig.update_xaxes(gridcolor="#2a2a40", tickfont=dict(size=8))
    fig.update_yaxes(gridcolor="#2a2a40", tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### LSA — SVD Variance Explained")
    cumvar = svd.explained_variance_ratio_.cumsum()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Scree Plot (50 components)", "Cumulative Variance"])
    fig.add_trace(go.Scatter(x=list(range(1,51)), y=svd.explained_variance_ratio_[:50].tolist(),
                             mode="lines+markers", line=dict(color="#7c6aff", width=2),
                             fill="tozeroy", marker=dict(size=4),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1,151)), y=cumvar.tolist(),
                             mode="lines", line=dict(color="#ff6a9e", width=2),
                             showlegend=False), row=1, col=2)
    for thr, clr, lbl in [(0.5,"#ffd06a","50%"),(0.8,"#6affd4","80%")]:
        fig.add_hline(y=thr, line_color=clr, line_dash="dash", line_width=1, row=1, col=2,
                      annotation_text=lbl, annotation_font_size=9)
    fig.update_layout(**PLOTLY_THEME, height=340, margin=dict(l=40,r=40,t=60,b=40))
    fig.update_xaxes(gridcolor="#2a2a40", linecolor="#2a2a40")
    fig.update_yaxes(gridcolor="#2a2a40", linecolor="#2a2a40")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"150 SVD components explain {cumvar[-1]:.1%} of total TF-IDF variance")

    st.markdown("### Structural Features — Mean per Class")
    feat_labels = ["Links","Forms","Inputs","Tables","Images","Headings","List Items","Text Len"]
    fig = make_subplots(rows=2, cols=4, subplot_titles=feat_labels,
                        horizontal_spacing=0.06, vertical_spacing=0.14)
    for i, (feat, lbl) in enumerate(zip(STRUCT_COLS, feat_labels)):
        row, col = divmod(i, 4)
        means = [train_df[train_df["label"]==c][feat].mean() for c in CLASSES]
        fig.add_trace(go.Bar(x=CLASSES, y=means, marker_color=PALETTE,
                             marker_line_color="#0a0a0f", name=lbl, showlegend=False),
                      row=row+1, col=col+1)
    fig.update_layout(**PLOTLY_THEME, height=520, margin=dict(l=10,r=10,t=60,b=20),
                      title="Mean Structural HTML Features per Class")
    fig.update_xaxes(gridcolor="#2a2a40", tickfont=dict(size=8), tickangle=30)
    fig.update_yaxes(gridcolor="#2a2a40", tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODELS
# ════════════════════════════════════════════════════════════════════════════════
with tab_models:
    col1, col2 = st.columns(2)
    for col, name, color, report, acc in [
        (col1, "KNN", "#7c6aff", knn_report, knn_acc),
        (col2, "SVM", "#ff6a9e", svm_report, svm_acc),
    ]:
        wa = report["weighted avg"]
        col.markdown(f"""<div style="background:#ffffff;border:1px solid #e4e6ec;border-top:3px solid {color};padding:1.2rem;margin-bottom:1rem;border-radius:10px;">
          <div style="font-size:.65rem;color:#888aa0;text-transform:uppercase;letter-spacing:.1em">{name}</div>
          <div style="font-family:Inter,sans-serif;font-size:2.5rem;font-weight:700;color:{color}">{acc * 100:.1f}%</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;margin-top:.8rem">
            <div style="background:#f7f8fa;padding:.6rem;border:1px solid #e4e6ec;border-radius:6px"><div style="font-size:.6rem;color:#888aa0">PRECISION</div><div style="font-size:1rem;font-weight:600;color:#1a1a2e">{wa['precision'] * 100:.1f}%</div></div>
            <div style="background:#f7f8fa;padding:.6rem;border:1px solid #e4e6ec;border-radius:6px"><div style="font-size:.6rem;color:#888aa0">RECALL</div><div style="font-size:1rem;font-weight:600;color:#1a1a2e">{wa['recall'] * 100:.1f}%</div></div>
            <div style="background:#f7f8fa;padding:.6rem;border:1px solid #e4e6ec;border-radius:6px"><div style="font-size:.6rem;color:#888aa0">F1-SCORE</div><div style="font-size:1rem;font-weight:600;color:#1a1a2e">{wa['f1-score'] * 100:.1f}%</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Confusion Matrices")
    col1, col2 = st.columns(2)
    for col, name, cm, cmap in [(col1,"KNN",knn_cm,"Blues"),(col2,"SVM",svm_cm,"RdPu")]:
        fig = go.Figure(go.Heatmap(z=cm, x=CLASSES, y=CLASSES, colorscale=cmap,
                                   text=cm.tolist(), texttemplate="%{text}",
                                   textfont=dict(size=10), showscale=True))
        fig.update_layout(title=f"{name} Confusion Matrix",
                          xaxis_title="Predicted", yaxis_title="True",
                          **PLOTLY_THEME, height=380, margin=dict(l=80,r=20,t=50,b=80))
        fig.update_xaxes(tickangle=30, tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=9))
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("### Per-Class F1 Score Comparison")
    knn_f1 = [knn_report[c]["f1-score"] for c in CLASSES]
    svm_f1 = [svm_report[c]["f1-score"] for c in CLASSES]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="KNN", x=CLASSES, y=knn_f1, marker_color="rgba(124,106,255,0.6)",
                         marker_line_width=1.5))
    fig.add_trace(go.Bar(name="SVM", x=CLASSES, y=svm_f1, marker_color="rgba(255,106,158,0.6)",
                         marker_line_width=1.5))
    fig.update_layout(barmode="group",
                      yaxis=dict(range=[0,1.05], gridcolor="#2a2a40"),
                      xaxis=dict(gridcolor="#2a2a40"), height=350, margin=dict(l=40,r=20,t=40,b=40),
                      legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Per-Class Classification Report")
    sel_model = st.radio("Select model:", ["KNN", "SVM"], horizontal=True)
    report = knn_report if sel_model == "KNN" else svm_report
    rows = [{"Class": cls, "Precision": f"{r['precision']*100:.1f}%",
             "Recall": f"{r['recall']*100:.1f}%", "F1-Score": f"{r['f1-score']*100:.1f}%",
             "Support": int(r["support"])}
            for cls in CLASSES for r in [report[cls]]]
    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    if sel_model == "SVM":
        st.markdown("### SVM Discriminative Coefficients per Class")
        fig = make_subplots(rows=2, cols=4, subplot_titles=CLASSES,
                            horizontal_spacing=0.06, vertical_spacing=0.14)
        for i, (cls, color) in enumerate(zip(CLASSES, PALETTE)):
            row, col = divmod(i, 4)
            coef = svm_model.coef_[i]
            top  = coef.argsort()[::-1][:10]
            fig.add_trace(go.Bar(x=coef[top][::-1].tolist(),
                                 y=[vocab[j] for j in top][::-1], orientation="h",
                                 name=cls, showlegend=False), row=row+1, col=col+1)
        fig.update_layout(**PLOTLY_THEME, height=560, margin=dict(l=10,r=10,t=60,b=20),
                          title="SVM Coefficients: Top Positive Features per Class")
        fig.update_xaxes(gridcolor="#2a2a40", tickfont=dict(size=8))
        fig.update_yaxes(gridcolor="#2a2a40", tickfont=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — TOP WORDS
# ════════════════════════════════════════════════════════════════════════════════
with tab_words:
    st.markdown("### Top Discriminative Terms per Class")
    st.caption("Term frequencies from training set (CountVectorizer, English stop-words removed)")
    selected = st.multiselect("Show classes:", CLASSES, default=CLASSES)
    r_idx, c_idx = 1, 1
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=[c for c in CLASSES if c in selected]+[""]*(8-len(selected)),
                        horizontal_spacing=0.06, vertical_spacing=0.14)
    for cls, color in zip(CLASSES, PALETTE):
        if cls not in selected: continue
        counts = word_counts[cls]
        top    = counts.argsort()[::-1][:14]
        fig.add_trace(go.Bar(x=counts[top][::-1].tolist(), y=[cv_vocab[j] for j in top][::-1],
                             orientation="h", marker_color=color,
                             marker_line_color="#0a0a0f", name=cls, showlegend=False),
                      row=r_idx, col=c_idx)
        c_idx += 1
        if c_idx > 4: c_idx, r_idx = 1, r_idx + 1
    fig.update_layout(**PLOTLY_THEME, height=560, margin=dict(l=10,r=10,t=60,b=20),
                      title="Top 14 Terms by Frequency per Class")
    fig.update_xaxes(gridcolor="#2a2a40", tickfont=dict(size=8))
    fig.update_yaxes(gridcolor="#2a2a40", tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Term Frequency Heatmap")
    top_terms = list(dict.fromkeys(
        cv_vocab[j] for cls in CLASSES for j in word_counts[cls].argsort()[::-1][:20]
    ))[:30]
    cv_vocab_list = list(cv_vocab)
    heatmap_data  = [[int(word_counts[cls][cv_vocab_list.index(t)]) if t in cv_vocab_list else 0
                      for t in top_terms] for cls in CLASSES]
    fig = go.Figure(go.Heatmap(z=heatmap_data, x=top_terms, y=CLASSES, colorscale="Viridis",
                               text=[[str(v) for v in row] for row in heatmap_data],
                               texttemplate="%{text}", textfont=dict(size=8)))
    fig.update_layout(title="Term Frequency Heatmap (Classes x Terms)", **PLOTLY_THEME,
                      height=380, margin=dict(l=80,r=20,t=50,b=80))
    fig.update_xaxes(tickangle=40, tickfont=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)
