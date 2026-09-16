import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import re
import os
from collections import Counter

st.set_page_config(
    page_title="CogniScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?
family=Inter:wght@300;400;500;600;700;800
&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(
        135deg,
        #EFF6FF 0%,
        #DBEAFE 40%,
        #E0F2FE 70%,
        #EDE9FE 100%);
    min-height: 100vh;
}
.hero-banner {
    background: linear-gradient(
        135deg,
        #0F2356 0%,
        #1B3A6B 45%,
        #1D9E75 100%);
    padding: 40px 44px;
    border-radius: 20px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 8px 32px
        rgba(15,35,86,0.25);
}
.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 16px;
    opacity: 0.88;
    font-weight: 300;
    margin: 0 0 20px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px;
}
.stat-card {
    background: white;
    border: 1px solid #DBEAFE;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 12px
        rgba(37,99,168,0.08);
}
.stat-val {
    font-size: 26px;
    font-weight: 800;
    margin: 0;
}
.stat-lbl {
    font-size: 11px;
    color: #6B7280;
    margin: 4px 0 0;
    font-weight: 500;
}
.result-card {
    border-radius: 16px;
    padding: 28px 32px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.result-AD {
    background: linear-gradient(
        135deg,#FEF2F2,#FECACA);
    border: 2px solid #C0392B;
}
.result-Control {
    background: linear-gradient(
        135deg,#ECFDF5,#A7F3D0);
    border: 2px solid #0F6E56;
}
.result-MCI {
    background: linear-gradient(
        135deg,#FFFBEB,#FDE68A);
    border: 2px solid #B05A00;
}
.section-card {
    background: white;
    border: 1px solid #DBEAFE;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 2px 12px
        rgba(37,99,168,0.08);
    margin-bottom: 16px;
}
.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #1B3A6B;
    margin: 0 0 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #EFF6FF;
}
.prob-track {
    background: #E5F0FF;
    border-radius: 99px;
    height: 14px;
    overflow: hidden;
    margin: 4px 0 12px;
}
.prob-fill-AD {
    height: 100%;
    background: linear-gradient(
        90deg,#C0392BAA,#C0392B);
    border-radius: 99px;
}
.prob-fill-Control {
    height: 100%;
    background: linear-gradient(
        90deg,#0F6E56AA,#0F6E56);
    border-radius: 99px;
}
.prob-fill-MCI {
    height: 100%;
    background: linear-gradient(
        90deg,#B05A00AA,#B05A00);
    border-radius: 99px;
}
.expl-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 10px 0;
    padding: 12px 16px;
    background: linear-gradient(
        135deg,#F0F7FF,#EFF6FF);
    border-radius: 10px;
}
.expl-rank {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}
.cds-track {
    background: linear-gradient(
        90deg,
        #1D9E75 0%,
        #E67E22 50%,
        #C0392B 100%);
    border-radius: 99px;
    height: 16px;
    margin: 8px 0;
    position: relative;
}
.disclaimer {
    background: linear-gradient(
        135deg,#FFFBEB,#FEF3C7);
    border: 1px solid #FCD34D;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 12px;
    color: #78350F;
    margin-top: 20px;
}
div[data-testid="stButton"] button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"]
    button[kind="primary"] {
    background: linear-gradient(
        135deg,#1B3A6B,#2563A8) !important;
    border: none !important;
    font-size: 16px !important;
    padding: 14px 28px !important;
    box-shadow: 0 4px 14px
        rgba(37,99,168,0.35) !important;
}
footer { display: none !important; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le     = joblib.load("model/label_encoder.pkl")
    with open("model/features.json") as f:
        features = json.load(f)
    return model, scaler, le, features

model, scaler, le, FEATURES = load_model()

FEAT_LABELS = {
    'mattr'           : 'Vocabulary diversity',
    'filler_ratio'    : 'Filler word usage',
    'repetition_ratio': 'Word repetition',
    'avg_sent_len'    : 'Sentence length',
    'content_ratio'   : 'Content word ratio',
    'pronoun_ratio'   : 'Pronoun usage',
    'article_ratio'   : 'Article usage',
    'conj_ratio'      : 'Conjunction usage',
    'sub_ratio'       : 'Sentence complexity',
    'utterance_count' : 'Utterance count',
    'pause_ratio'     : 'Pause ratio',
    'speech_rate'     : 'Speech rate',
    'energy_mean'     : 'Vocal energy',
    'pitch_mean'      : 'Voice pitch',
    'centroid_mean'   : 'Spectral centroid',
    'cds'             : 'Cognitive Drift Score ★',
}

STOPWORDS = {
    'the','a','an','and','or','but','in','on',
    'at','to','for','of','with','by','from','is',
    'are','was','were','be','been','has','have',
    'had','do','does','did','will','would','could',
    'should','may','might','shall','can','not','no',
    'so','yet','as','if','then','that','this',
    'these','those','just','very','too','some',
    'such','than','both','each','few','more','most'
}
FILLERS      = {'um','uh','mhm','hmm','er','ah'}
PRONOUNS     = {'i','me','my','he','him','his',
                'she','her','it','we','us','they',
                'them','you','this','that'}
ARTICLES     = {'the','a','an'}
CONJUNCTIONS = {'and','but','or','nor','so',
                'although','though','because',
                'since','while','whereas'}
SUBORDINATORS= {'because','although','though',
                'while','when','after','before',
                'since','if','unless','until',
                'where','whether','who','which'}

def compute_mattr(words, window=50):
    N = len(words)
    if N < window:
        return len(set(words))/N if N>0 else 0.0
    return float(np.mean([
        len(set(words[i:i+window]))/window
        for i in range(N-window+1)]))

def extract_features(transcript, cds_val=50.0):
    text  = transcript.lower().strip()
    sents = [s.strip() for s in
             re.split(r'[.!?]',text)
             if len(s.strip())>2]
    words = re.findall(r"\b[a-z']+\b",text)
    N = len(words)
    if N < 5:
        return None, None
    V  = len(set(words))
    wf = Counter(words)
    ling = {
        'mattr'           : compute_mattr(words),
        'filler_ratio'    : sum(
            wf.get(f,0) for f in FILLERS)/N,
        'repetition_ratio': sum(
            1 for w,c in wf.items()
            if c>2)/V if V>0 else 0.0,
        'avg_sent_len'    : float(np.mean(
            [len(s.split()) for s in sents]))
            if sents else 0.0,
        'content_ratio'   : len(
            [w for w in words
             if w not in STOPWORDS])/N,
        'pronoun_ratio'   : sum(
            wf.get(p,0) for p in PRONOUNS)/N,
        'article_ratio'   : sum(
            wf.get(a,0) for a in ARTICLES)/N,
        'conj_ratio'      : sum(
            wf.get(c,0) for c in CONJUNCTIONS)/N,
        'sub_ratio'       : sum(
            wf.get(s,0) for s in SUBORDINATORS)/N,
        'utterance_count' : float(len(sents)),
    }
    acou = {
        'pause_ratio'  : 0.50,
        'speech_rate'  : 0.12,
        'energy_mean'  : 0.045,
        'pitch_mean'   : 170.0,
        'centroid_mean': 1500.0,
    }
    feat_dict = {**ling, **acou, 'cds': cds_val}
    feat_vec  = np.array(
        [feat_dict.get(f, 0.0)
         for f in FEATURES]).reshape(1,-1)
    return feat_dict, feat_vec

# ── Hero Banner ────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🧠 CogniScan AI</p>
    <p class="hero-sub">
        Explainable multimodal three-stage
        cognitive decline detection from speech
    </p>
    <span class="hero-badge">
        🎯 AD · MCI · Control
    </span>
    <span class="hero-badge">
        🔍 Dual-Layer XAI
    </span>
    <span class="hero-badge">
        📈 Cognitive Drift Score
    </span>
    <span class="hero-badge">
        ⚡ F1=0.670 · AUC=0.851
    </span>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-val"
           style="color:#1B3A6B">0.670</p>
        <p class="stat-lbl">Weighted F1</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-val"
           style="color:#1B3A6B">0.851</p>
        <p class="stat-lbl">Macro AUC</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-val"
           style="color:#C0392B">0.962</p>
        <p class="stat-lbl">AD Detection AUC</p>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-val"
           style="color:#1B3A6B">326</p>
        <p class="stat-lbl">
            DementiaBank Patients
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Section ──────────────────────────────────────
st.markdown("""
<div class="section-card">
    <p class="section-title">
        📋 Patient Transcript
    </p>
</div>""", unsafe_allow_html=True)

# Sample buttons
sb1,sb2,sb3,_ = st.columns([1,1,1,3])
with sb1:
    ad_btn = st.button(
        "🔴 AD sample",
        use_container_width=True)
with sb2:
    mci_btn = st.button(
        "🟡 MCI sample",
        use_container_width=True)
with sb3:
    con_btn = st.button(
        "🟢 Control sample",
        use_container_width=True)

AD_SAMPLE = """The woman is, um, washing the dishes. And the, uh, boy is, um, the boy is getting cookies. Um. The stool is falling. Um, the woman, the woman is washing. And, um, water. Water is, uh, going. The boy, the boy, um, cookies. Um. The woman is, the woman, um."""

MCI_SAMPLE = """There is a woman washing dishes at the sink. The water seems to be overflowing. A boy is, um, trying to get cookies from the cabinet. He is standing on a stool. The stool looks like it might fall. There is a, uh, a window behind the woman."""

CONTROL_SAMPLE = """In this picture I can see a kitchen scene. A woman is washing dishes at the sink while water overflows onto the floor. She seems distracted. A young boy is standing on a stool reaching into a cookie jar above. A girl is standing nearby watching him. Through the window outside you can see a pleasant sunny day."""

if ad_btn:
    st.session_state['transcript'] = AD_SAMPLE
if mci_btn:
    st.session_state['transcript'] = MCI_SAMPLE
if con_btn:
    st.session_state['transcript'] = CONTROL_SAMPLE

transcript = st.text_area(
    "Transcript",
    value=st.session_state.get(
        'transcript', ''),
    placeholder=(
        "Paste the Cookie Theft picture "
        "description transcript here..."),
    height=160,
    label_visibility="collapsed")

cds_val = st.slider(
    "📈 Cognitive Drift Score"
    "  —  50 = stable baseline"
    "  ·  higher = more cognitive drift",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.5)

st.markdown("<br>", unsafe_allow_html=True)
analyse = st.button(
    "🔍  Analyse Patient",
    type="primary",
    use_container_width=True)

# ── Analysis ───────────────────────────────────────────
if analyse:
    if not transcript or \
            len(transcript.strip()) < 20:
        st.error(
            "Please enter a transcript of "
            "at least 20 characters.")
        st.stop()

    with st.spinner(
            "Analysing speech patterns..."):

        feat_dict, feat_vec = \
            extract_features(transcript, cds_val)

        if feat_dict is None:
            st.error("Transcript too short.")
            st.stop()

        feat_scaled = scaler.transform(feat_vec)
        probs       = model.predict_proba(
            feat_scaled)[0]
        pred_idx    = int(np.argmax(probs))
        pred_class  = le.classes_[pred_idx]
        confidence  = probs[pred_idx]

        shap_vals = np.zeros(len(FEATURES))
        try:
            import shap
            rf_comp   = model.estimators_[0]
            explainer = shap.TreeExplainer(
                rf_comp)
            sv_raw    = explainer.shap_values(
                feat_scaled)
            sv_arr    = np.array(sv_raw)
            if sv_arr.ndim == 3:
                if sv_arr.shape[2] == 3:
                    shap_vals = \
                        sv_arr[0,:,pred_idx]
                else:
                    shap_vals = \
                        sv_arr[pred_idx,0,:]
            elif isinstance(sv_raw, list):
                shap_vals = sv_raw[pred_idx][0]
            shap_vals = np.array(
                shap_vals).flatten()
            if len(shap_vals) != len(FEATURES):
                shap_vals = np.zeros(
                    len(FEATURES))
        except Exception:
            pass

    # ── Results ────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "## 📊 Analysis Results",
        unsafe_allow_html=False)

    cls_color = {
        'AD'     : '#C0392B',
        'Control': '#0F6E56',
        'MCI'    : '#B05A00'}
    cls_icon = {
        'AD'     : '🔴',
        'Control': '🟢',
        'MCI'    : '🟡'}
    cls_desc = {
        'AD'     : ("Alzheimer's Disease patterns "
                    "detected. Speech shows "
                    "significant acoustic and "
                    "linguistic decline consistent"
                    " with established AD."),
        'Control': ("Healthy Control patterns "
                    "detected. Speech is consistent"
                    " with normal cognitive aging. "
                    "Continue routine monitoring."),
        'MCI'    : ("Mild Cognitive Impairment "
                    "detected. Speech shows "
                    "early-stage decline patterns."
                    " Clinical follow-up "
                    "recommended.")}

    color = cls_color[pred_class]
    icon  = cls_icon[pred_class]

    # Prediction card
    st.markdown(f"""
    <div class="result-card result-{pred_class}">
        <div style="font-size:11px;
                    font-weight:700;
                    color:{color};
                    text-transform:uppercase;
                    letter-spacing:1.5px;
                    margin-bottom:10px;">
            Diagnostic Prediction
        </div>
        <div style="font-size:36px;
                    font-weight:800;
                    color:{color};
                    margin-bottom:8px;">
            {icon} {pred_class}
        </div>
        <div style="display:inline-block;
                    background:{color};
                    color:white;
                    font-size:13px;
                    font-weight:600;
                    padding:4px 16px;
                    border-radius:20px;
                    margin-bottom:12px;">
            {confidence:.1%} confidence
        </div>
        <div style="font-size:14px;
                    color:#374151;
                    line-height:1.7;">
            {cls_desc[pred_class]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns — probs + explanation
    col_l, col_r = st.columns(2)

    with col_l:
        # Probability bars
        order = np.argsort(probs)[::-1]
        bars_html = ""
        for i in order:
            cls  = le.classes_[i]
            prob = probs[i]
            pct  = int(prob * 100)
            c    = cls_color[cls]
            is_p = cls == pred_class
            bars_html += f"""
            <div style="margin:10px 0;">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    font-size:13px;
                    font-weight:{'700'
                        if is_p else '500'};
                    margin-bottom:5px;
                    color:#1F2937;">
                    <span>
                        {cls_icon[cls]}
                        {cls}
                        {'✓' if is_p else ''}
                    </span>
                    <span style="color:{c};
                                 font-weight:700;">
                        {prob:.1%}
                    </span>
                </div>
                <div class="prob-track">
                    <div style="
                        width:{pct}%;
                        height:100%;
                        background:linear-gradient(
                            90deg,{c}99,{c});
                        border-radius:99px;">
                    </div>
                </div>
            </div>"""

        st.markdown(f"""
        <div class="section-card">
            <p class="section-title">
                📊 Class Probabilities
            </p>
            {bars_html}
        </div>""", unsafe_allow_html=True)

    with col_r:
        # Layer 2 explanation
        top_idx = np.argsort(
            np.abs(shap_vals))[::-1][:3]
        items_html = ""
        for rank,idx in enumerate(top_idx,1):
            fname = FEATURES[idx]
            fval  = feat_scaled[0, idx]
            sval  = shap_vals[idx]
            direction = ('↑ elevated'
                         if fval > 0
                         else '↓ reduced')
            impact    = ('increases'
                         if sval > 0
                         else 'decreases')
            name  = FEAT_LABELS.get(
                fname, fname)
            d_col = ('#C0392B'
                     if sval > 0
                     else '#0F6E56')
            items_html += f"""
            <div style="
                display:flex;
                align-items:flex-start;
                gap:12px;margin:10px 0;
                padding:12px 16px;
                background:linear-gradient(
                    135deg,#F0F7FF,#EFF6FF);
                border-radius:10px;
                border-left:4px solid {color};">
                <div style="
                    background:{color};
                    color:white;
                    font-size:11px;
                    font-weight:700;
                    width:24px;height:24px;
                    border-radius:50%;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    flex-shrink:0;">
                    {rank}
                </div>
                <div>
                    <div style="
                        font-size:13px;
                        font-weight:600;
                        color:#1B3A6B;">
                        {name}
                        <span style="
                            font-weight:400;
                            color:#6B7280;
                            font-size:12px;">
                            is {direction}
                        </span>
                    </div>
                    <div style="
                        font-size:12px;
                        color:{d_col};
                        font-weight:600;
                        margin-top:3px;">
                        {impact}
                        {pred_class} likelihood
                    </div>
                </div>
            </div>"""

        st.markdown(f"""
        <div class="section-card">
            <p class="section-title">
                💬 Patient Explanation
                <span style="
                    font-size:11px;
                    font-weight:400;
                    color:#9CA3AF;">
                    Top 3 SHAP factors
                </span>
            </p>
            {items_html}
            <div style="
                font-size:11px;
                color:#9CA3AF;
                font-style:italic;
                margin-top:12px;
                padding-top:10px;
                border-top:1px solid #EFF6FF;">
                TreeSHAP on RF component.
                Research use only.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CDS Section
    cds_v = float(cds_val)
    if cds_v < 48:
        cds_status = "🟢 Stable"
        cds_color  = "#0F6E56"
        cds_note   = ("CDS is near or below the "
                      "Control population mean "
                      "(μ=46.0). No significant "
                      "cognitive drift detected.")
    elif cds_v < 55:
        cds_status = "🟡 Borderline"
        cds_color  = "#B05A00"
        cds_note   = ("CDS falls between Control "
                      "(μ=46.0) and MCI (μ=51.6) "
                      "population means. Monitoring"
                      " is recommended.")
    else:
        cds_status = "🔴 Elevated drift"
        cds_color  = "#C0392B"
        cds_note   = ("CDS exceeds the MCI "
                      "population mean (μ=51.6). "
                      "Significant longitudinal "
                      "cognitive drift detected.")

    ctrl_pct = 46
    mci_pct  = 51.6
    pat_pct  = min(cds_v, 100)

    st.markdown(f"""
    <div class="section-card">
        <p class="section-title">
            📈 Cognitive Drift Score (CDS)
            <span style="
                font-size:11px;
                font-weight:400;
                color:#9CA3AF;">
                Longitudinal biomarker trajectory
            </span>
        </p>
        <div style="display:flex;
                    align-items:baseline;
                    gap:12px;
                    margin-bottom:12px;">
            <span style="font-size:36px;
                         font-weight:800;
                         color:{cds_color};">
                {cds_v:.1f}
            </span>
            <span style="font-size:14px;
                         color:#6B7280;">
                / 100
            </span>
            <span style="
                font-size:14px;
                font-weight:700;
                color:{cds_color};">
                {cds_status}
            </span>
        </div>
        <div style="
            position:relative;
            background:#E5F0FF;
            border-radius:99px;
            height:20px;
            margin:8px 0 4px;
            overflow:hidden;">
            <div style="
                position:absolute;
                left:0;top:0;
                width:{pat_pct}%;
                height:100%;
                background:linear-gradient(
                    90deg,
                    #1D9E75,#E67E22,#C0392B);
                border-radius:99px;
                opacity:0.85;">
            </div>
            <div style="
                position:absolute;
                left:{ctrl_pct}%;
                top:0;width:2px;
                height:100%;
                background:white;
                opacity:0.9;">
            </div>
            <div style="
                position:absolute;
                left:{mci_pct}%;
                top:0;width:2px;
                height:100%;
                background:white;
                opacity:0.9;">
            </div>
        </div>
        <div style="
            display:flex;
            justify-content:space-between;
            font-size:10px;
            color:#6B7280;
            margin-bottom:16px;">
            <span>0</span>
            <span style="
                color:#0F6E56;
                font-weight:600;">
                Control μ=46.0
            </span>
            <span style="
                color:#B05A00;
                font-weight:600;">
                MCI μ=51.6
            </span>
            <span>100</span>
        </div>
        <div style="
            font-size:13px;
            color:#374151;
            line-height:1.6;
            padding:12px 16px;
            background:#F0F7FF;
            border-radius:8px;
            border-left:3px solid {cds_color};">
            {cds_note}
        </div>
        <div style="
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:12px;
            margin-top:16px;">
            <div style="text-align:center;
                        padding:10px;
                        background:#ECFDF5;
                        border-radius:8px;">
                <div style="font-weight:700;
                            color:#0F6E56;">
                    46.0</div>
                <div style="font-size:11px;
                            color:#6B7280;">
                    🟢 Control mean</div>
            </div>
            <div style="text-align:center;
                        padding:10px;
                        background:#FFFBEB;
                        border-radius:8px;">
                <div style="font-weight:700;
                            color:#B05A00;">
                    51.6</div>
                <div style="font-size:11px;
                            color:#6B7280;">
                    🟡 MCI mean</div>
            </div>
            <div style="text-align:center;
                        padding:10px;
                        background:#EFF6FF;
                        border-radius:8px;">
                <div style="font-weight:700;
                            color:#1B3A6B;">
                    {cds_v:.1f}</div>
                <div style="font-size:11px;
                            color:#6B7280;">
                    👤 This patient</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature table
    with st.expander(
            "🔬 View extracted speech features"):
        feat_rows = []
        for f in FEATURES:
            modality = (
                'CDS' if f=='cds'
                else 'Acoustic'
                if f in [
                    'pause_ratio','speech_rate',
                    'energy_mean','pitch_mean',
                    'centroid_mean']
                else 'Linguistic')
            feat_rows.append({
                'Feature': FEAT_LABELS.get(f,f),
                'Modality': modality,
                'Value': round(
                    feat_dict.get(f,0.0),4),
            })
        st.dataframe(
            pd.DataFrame(feat_rows),
            hide_index=True,
            use_container_width=True)
        st.caption(
            "Acoustic features use population "
            "medians (text-only demo). "
            "Upload audio for real values.")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Research demonstration only.</strong>
    This tool is not a substitute for clinical
    diagnosis. Acoustic features use population
    medians in this text-only demo. Model:
    RF+GB+SVM soft-voting ensemble trained on
    DementiaBank Pitt Corpus (N=326).
    Weighted F1=0.670, Macro AUC=0.851.
</div>
""", unsafe_allow_html=True)
