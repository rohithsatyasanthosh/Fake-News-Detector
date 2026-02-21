import streamlit as st
import joblib
import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import streamlit.components.v1 as components
from src.training import FakeNewsModel
from src.explain import NewsExplainer

# Configure Page
st.set_page_config(
    page_title="Veritas | Professional News Auditor",
    page_icon="⚖️",
    layout="wide",
)

# Robust NLTK Asset Management
@st.cache_resource
def setup_nltk():
    assets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for asset in assets:
        try:
            nltk.download(asset, quiet=True)
        except Exception as e:
            st.error(f"NLTK Error: {e}")

setup_nltk()

# --- STYLING (User-Friendly Professional) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lexend:wght@400;700&display=swap');
    
    :root {
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --secondary: #64748b;
        --bg: #f8fafc;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --border: #e2e8f0;
        --success: #166534;
        --error: #991b1b;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    h1, h2, h3 { font-family: 'Lexend', sans-serif !important; color: var(--text); }
    * { font-family: 'Inter', sans-serif; }

    /* Layout Containers */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Clean Card Component */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    /* Hero Branding */
    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    .brand-subtitle {
        color: var(--muted);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Input Area */
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        font-size: 1rem !important;
    }

    /* Primary Action Button */
    .stButton > button {
        background-color: var(--primary);
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2.5rem;
        transition: all 0.2s;
        border: none;
    }
    .stButton > button:hover {
        background-color: var(--primary-hover);
        transform: translateY(-1px);
    }

    /* Result Indicators */
    .verdict {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }
    .verdict-real { background-color: #dcfce7; color: var(--success); }
    .verdict-fake { background-color: #fee2e2; color: var(--error); }

    /* Empty State / Help Cards */
    .help-card {
        background-color: #eff6ff;
        border-left: 4px solid var(--primary);
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .help-title { font-weight: 700; color: var(--primary); margin-bottom: 0.25rem; }
    .help-text { color: #1e40af; font-size: 0.9rem; }

    /* Sidebar Navigation */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# --- ENGINE LOADING ---
@st.cache_resource
def load_engine():
    if not os.path.exists('models/model.pkl'):
        return None, None
    try:
        model = joblib.load('models/model.pkl')
        vec = joblib.load('models/vectorizer.pkl')
        return model, vec
    except:
        return None, None

@st.cache_data
def load_dataset():
    if os.path.exists('data/raw/True.csv') and os.path.exists('data/raw/Fake.csv'):
        try:
            df_true = pd.read_csv('data/raw/True.csv')
            df_fake = pd.read_csv('data/raw/Fake.csv')
            df_true['target'] = 1
            df_fake['target'] = 0
            df = pd.concat([df_true, df_fake]).reset_index(drop=True)
            df['total_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
            return df
        except:
            return None
    return None

@st.cache_resource
def get_explainer(_model, _vec):
    return NewsExplainer(_model, _vec)

model_obj, vec_obj = load_engine()
data_df = load_dataset()

# Sidebar
with st.sidebar:
    st.markdown('<div style="color:var(--primary); font-size: 1.8rem; font-weight: 800; font-family:Lexend;">VERITAS</div>', unsafe_allow_html=True)
    st.markdown("Professional News Verification")
    st.divider()
    
    navigation = st.radio(
        "Navigation",
        ["Audit Dashboard", "Intelligence Tracker"],
        index=0
    )
    
    st.divider()
    if model_obj:
        st.success("Analysis Engine Online")
    else:
        st.error("Engine Offline - Run Train.py")

# --- DASHBOARD PAGE ---
if navigation == "Audit Dashboard":
    st.markdown('<div class="brand-title">Neural Verification Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Forensic AI analysis for identifying disinformation and linguistic anomalies.</div>', unsafe_allow_html=True)

    # Populate Empty Dashboard with Intelligence context
    if "probs" not in st.session_state:
        st.markdown('<div class="help-card">', unsafe_allow_html=True)
        st.markdown("### 🧬 Welcome to Veritas XAI")
        st.markdown("""
        Our core intelligence engine uses **Transfer Learning & TF-IDF Vectorization** to cross-reference news content 
        against a database of **44,000 verified and flagged records**. 
        
        **To begin:** Paste an article or social media post in the field below. Our internal 'Linguistic Agent' 
        will then scan for sensationalism, speculation, and pattern matching.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    col_input, col_results = st.columns([1.5, 1], gap="large")

    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖋️ News Stream Input")
        user_input = st.text_area(
            "Paste Content",
            placeholder="Input news sequence for neural audit...",
            height=300,
            label_visibility="collapsed",
            key="input_area_final"
        )
        submit = st.button("INITIATE NEURAL SCAN")
        st.markdown('</div>', unsafe_allow_html=True)

        if submit:
            if not model_obj:
                st.error("Engine Offline.")
            elif len(user_input.strip().split()) < 3:
                st.warning("Insufficient signal strength (min 3 words).")
            else:
                with st.spinner("Processing Semantic Trace..."):
                    fn = FakeNewsModel()
                    fn.model = model_obj
                    fn.vectorizer = vec_obj
                    probs = fn.predict(user_input)
                    st.session_state.probs = probs
                    st.session_state.last_text = user_input
                    
                    try:
                        explainer = get_explainer(model_obj, vec_obj)
                        st.session_state.shap_vals = explainer.get_local_explanation([user_input])
                        st.session_state.audit = explainer.get_linguistic_audit(user_input)
                    except:
                        st.session_state.shap_vals = None

        if "shap_vals" in st.session_state and st.session_state.shap_vals is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Linguistic Neuron Trace")
            shap_html = shap.plots.text(st.session_state.shap_vals[0], display=False)
            components.html(shap_html, height=250, scrolling=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        if "probs" in st.session_state:
            fake_p, real_p = st.session_state.probs
            audit = st.session_state.audit
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if real_p > 0.5:
                st.markdown('<div class="verdict verdict-real">✅ CERTIFIED ACCURATE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict verdict-fake">🚨 ANOMALY DETECTED</div>', unsafe_allow_html=True)
            
            st.metric("Credibility Signature", f"{real_p:.1%}")
            st.metric("Deceptive Match", f"{fake_p:.1%}")
            
            st.divider()
            st.markdown(f"**AI Agent Assessment:** {audit['assessment']}")
            for n, v in audit['stats'].items():
                st.progress(float(v.strip('%'))/100, text=f"{n}: {v}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card" style="text-align: center; opacity: 0.7;">', unsafe_allow_html=True)
            st.markdown("### 📡 Engine Status: Awaiting Input")
            st.write("Neutral assessment results will appear here after the scan is triggered.")
            st.image("https://img.icons8.com/ios-filled/512/2563eb/radar.png", width=120)
            st.markdown('</div>', unsafe_allow_html=True)

# --- RELATED ARCHIVES PAGE (TRACKER) ---
elif navigation == "Intelligence Tracker":
    st.markdown('<div class="brand-title">Intelligence Archive Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Synchronizing with our 44,000+ record forensic news database.</div>', unsafe_allow_html=True)

    if "last_text" not in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("Input news in the Dashboard to synchronize related intelligence.")
        st.write("""
        **How this works:** Our system creates a 'Semantic Vector' of your input and hunts for similar entries in our archives.
        By finding related news, we can determine if your story is part of a recurring disinformation pattern or a verified 
        news cycle from agencies like Reuters.
        """)
        st.image("https://img.icons8.com/fluency/512/search-in-cloud.png", width=200)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.spinner("QUERYING FORENSIC ARCHIVES..."):
            explainer = get_explainer(model_obj, vec_obj)
            related = explainer.get_related_intel(st.session_state.last_text, data_df)
            
            if not related:
                st.warning("No semantic signatures match your input in our database.")
            else:
                st.markdown(f"### Found {len(related)} Related News Signatures")
                
                # Dynamic Layout for Related News
                for res in related:
                    with st.container():
                        st.markdown(f"""
                        <div class="card" style="border-left: 5px solid {'#22c55e' if res['label'] == 'REAL' else '#ef4444'};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin:0; font-size:1.2rem; color:#1e293b;">{res['title']}</h3>
                                <span style="background:{'#dcfce7' if res['label'] == 'REAL' else '#fee2e2'}; 
                                      color:{'#166534' if res['label'] == 'REAL' else '#991b1b'}; 
                                      padding: 0.3rem 1rem; border-radius: 5px; font-weight: 800; font-size: 0.7rem;">
                                    {res['label']} CONTENT
                                </span>
                            </div>
                            <p style="color:#64748b; font-size:0.95rem; margin: 1rem 0; line-height:1.6;">{res['content']}</p>
                            <div style="display: flex; gap: 2rem; border-top: 1px solid #f1f5f9; padding-top: 0.8rem;">
                                <span style="font-size: 0.8rem; color:#2563eb;"><strong>SYNC MATCH:</strong> {res['similarity']}</span>
                                <span style="font-size: 0.8rem; color:#64748b;"><strong>Source Type:</strong> Verified Archive</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: var(--muted); border-top: 1px solid var(--border); padding-top: 1.5rem;">
    Veritas News Verification Engine | 2026 | Powered by XAI
</div>
""", unsafe_allow_html=True)
