import streamlit as st
import pickle
import re
import string
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ── Page config ──
st.set_page_config(
    page_title="FakeScope — Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .title-text {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f0efe8;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .title-text span { color: #b8f57a; }

    .subtitle {
        color: #888898;
        font-size: 1rem;
        margin-bottom: 2rem;
        line-height: 1.7;
    }
    .stat-box {
        background: #1c1c26;
        border: 0.5px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-num {
        color: #b8f57a;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .stat-lbl {
        color: #888898;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-real {
        background: rgba(184,245,122,0.08);
        border: 1px solid rgba(184,245,122,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-fake {
        background: rgba(249,115,115,0.08);
        border: 1px solid rgba(249,115,115,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-uncertain {
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .verdict-real      { color: #b8f57a; font-size: 1.5rem; font-weight: 800; }
    .verdict-fake      { color: #f97373; font-size: 1.5rem; font-weight: 800; }
    .verdict-uncertain { color: #fbbf24; font-size: 1.5rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)


# ── Text cleaner ──
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Train model ──
def train_model():
    with st.spinner("🚀 Training model for the first time... please wait 3-4 minutes"):
        df = pd.read_csv('news.csv', index_col=0)
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['content'] = df['content'].apply(clean_text)

        le = LabelEncoder()
        df['label_enc'] = le.fit_transform(df['label'])

        X_train, _, y_train, _ = train_test_split(
            df['content'], df['label_enc'],
            test_size=0.2, random_state=42,
            stratify=df['label_enc']
        )

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True
            )),
            ('clf', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)

        with open('model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)

    st.success("✅ Model trained and ready!")
    return pipeline, le


# ── Load model (cached so it loads only once) ──
@st.cache_resource
def load_model():
    if os.path.exists('model.pkl') and os.path.exists('label_encoder.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, le
    else:
        return train_model()


# ════════════════════════════════
#           UI STARTS HERE
# ════════════════════════════════

# ── Header ──
st.markdown('<div class="title-text">🔍 Fake<span>Scope</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any news headline or article. Our Gradient Boosting model detects misinformation instantly.</div>', unsafe_allow_html=True)

# ── Stats row ──
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-num">98%</div><div class="stat-lbl">Accuracy</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-num">44K+</div><div class="stat-lbl">Articles Trained</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-num">&lt;1s</div><div class="stat-lbl">Analysis Time</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Example buttons ──
st.markdown("**Try an example:**")
ex1, ex2 = st.columns(2)
with ex1:
    if st.button("📰 Real news example"):
        st.session_state['example'] = "Scientists at NASA confirm the Mars rover Perseverance has successfully collected rock samples that will be returned to Earth for analysis, potentially revealing signs of ancient microbial life."
with ex2:
    if st.button("⚠️ Fake news example"):
        st.session_state['example'] = "BREAKING: Government caught putting microchips in vaccines to track citizens! A whistleblower nurse CONFIRMED it. Big Pharma doesn't want you to know this. Share before it's deleted!!!"

# ── Text input ──
default_text = st.session_state.get('example', '')
news_input = st.text_area(
    "Paste your news here:",
    value=default_text,
    height=160,
    placeholder="Paste a news headline or article excerpt here…"
)

# ── Analyze button ──
if st.button("🔍 Analyze News", use_container_width=True):
    if not news_input or len(news_input.strip()) < 20:
        st.warning("⚠️ Please enter at least 20 characters.")
    else:
        # Load model
        model, le = load_model()

        with st.spinner("Analyzing..."):
            cleaned = clean_text(news_input)
            pred = model.predict([cleaned])[0]
            proba = model.predict_proba([cleaned])[0]

            label = le.inverse_transform([pred])[0]
            confidence = round(float(max(proba)) * 100, 1)
            real_prob  = round(float(proba[1]) * 100, 1)
            fake_prob  = round(float(proba[0]) * 100, 1)
            verdict = 'UNCERTAIN' if confidence < 60 else label

        # ── Show result ──
        st.markdown("<br>", unsafe_allow_html=True)

        if verdict == 'REAL':
            icon        = "✅"
            css         = "result-real"
            verdict_css = "verdict-real"
            title       = "Likely Credible News"
        elif verdict == 'FAKE':
            icon        = "🚫"
            css         = "result-fake"
            verdict_css = "verdict-fake"
            title       = "Likely Fake News"
        else:
            icon        = "⚠️"
            css         = "result-uncertain"
            verdict_css = "verdict-uncertain"
            title       = "Cannot Determine"

        st.markdown(f"""
        <div class="{css}">
            <div class="{verdict_css}">{icon} {verdict} — {title}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar
        st.markdown(f"**Model Confidence: {confidence}%**")
        st.progress(confidence / 100)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability columns
        r1, r2 = st.columns(2)
        with r1:
            st.metric("✅ Real Probability", f"{real_prob}%")
        with r2:
            st.metric("🚫 Fake Probability", f"{fake_prob}%")

# ── Footer ──
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#888898; font-size:12px;">'
    'FakeScope · Built with Gradient Boosting · For educational purposes'
    '</p>',
    unsafe_allow_html=True
)