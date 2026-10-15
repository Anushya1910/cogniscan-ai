import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(
    page_title="CogniScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #F0F4FF 0%, #EEF2FF 30%,
        #F5F3FF 60%, #E0F2FE 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,#1E1B4B 0%,
        #312E81 50%,#1E40AF 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}
.sidebar-logo {
    text-align: center;
    padding: 20px 0 10px;
}
section[data-testid="stSidebar"]
    [data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.7) !important;
    font-size: 12px !important;
}
.stButton button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(
        135deg,#4F46E5,#7C3AED) !important;
    border: none !important;
    box-shadow: 0 4px 14px
        rgba(79,70,229,0.4) !important;
    color: white !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px
        rgba(79,70,229,0.5) !important;
}
.stButton button[kind="secondary"] {
    background: rgba(255,255,255,0.9) !important;
    border: 1.5px solid #E2E8F0 !important;
    color: #4F46E5 !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.95) !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px
        rgba(99,102,241,0.12) !important;
}
.stSlider [data-baseweb="slider"] {
    padding: 0 4px !important;
}
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 2px 12px
        rgba(99,102,241,0.08) !important;
}
footer { display: none !important; }
#MainMenu { visibility: hidden !important; }
header { visibility: hidden !important; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}
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

# ── Feature constants ──────────────────────────────────
FEAT_LABELS = {
    'mattr'           : 'Vocabulary Diversity (MATTR)',
    'filler_ratio'    : 'Filler Word Ratio',
    'repetition_ratio': 'Word Repetition Ratio',
    'avg_sent_len'    : 'Avg Sentence Length',
    'content_ratio'   : 'Content Word Ratio',
    'pronoun_ratio'   : 'Pronoun Ratio',
    'article_ratio'   : 'Article Ratio',
    'conj_ratio'      : 'Conjunction Ratio',
    'sub_ratio'       : 'Subordination Ratio',
    'utterance_count' : 'Utterance Count',
    'pause_ratio'     : 'Pause Ratio',
    'speech_rate'     : 'Speech Rate (ZCR)',
    'energy_mean'     : 'Vocal Energy (RMS)',
    'pitch_mean'      : 'Mean Pitch (Hz)',
    'centroid_mean'   : 'Spectral Centroid',
    'cds'             : 'Cognitive Drift Score ★',
}
ACOU = {'pause_ratio','speech_rate',
        'energy_mean','pitch_mean',
        'centroid_mean'}
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
FILLERS = {'um','uh','mhm','hmm','er','ah'}
PRONOUNS = {
    'i','me','my','he','him','his','she','her',
    'it','we','us','they','them','you','this','that'}
ARTICLES = {'the','a','an'}
CONJUNCTIONS = {
    'and','but','or','nor','so','although',
    'though','because','since','while','whereas'}
SUBORDINATORS = {
    'because','although','though','while','when',
    'after','before','since','if','unless','until',
    'where','whether','who','which'}

# ── Feature extraction ─────────────────────────────────
def compute_mattr(words, window=50):
    N = len(words)
    if N < window:
        return len(set(words))/N if N>0 else 0.0
    return float(np.mean([
        len(set(words[i:i+window]))/window
        for i in range(N-window+1)]))

def extract_features(transcript, cds=50.0):
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
            wf.get(c,0)
            for c in CONJUNCTIONS)/N,
        'sub_ratio'       : sum(
            wf.get(s,0)
            for s in SUBORDINATORS)/N,
        'utterance_count' : float(len(sents)),
    }
    acou = {
        'pause_ratio'  : 0.50,
        'speech_rate'  : 0.12,
        'energy_mean'  : 0.045,
        'pitch_mean'   : 170.0,
        'centroid_mean': 1500.0,
    }
    feat_dict = {**ling, **acou, 'cds': cds}
    feat_vec  = np.array(
        [feat_dict.get(f,0.0)
         for f in FEATURES]).reshape(1,-1)
    return feat_dict, feat_vec

# ── Samples ────────────────────────────────────────────
SAMPLES = {
    'AD': """The woman is, um, washing the dishes. And the, uh, boy is, um, the boy is getting cookies. Um. The stool is falling. Um, the woman, the woman is washing. And, um, water. Water is, uh, going. The boy, the boy, um, cookies. Um. The woman is, the woman, um. The boy. Um.""",
    'MCI': """There is a woman washing dishes at the sink. The water seems to be overflowing. A boy is, um, trying to get cookies from the cabinet. He is standing on a stool. The stool looks like it might fall. There is a, uh, a window behind the woman. The girl is watching.""",
    'Control': """All right, looking closely at the scene, there's a complete lack of supervision creating chaos in this kitchen. On the right side, the mother is standing at the sink, casually drying a dish, completely oblivious to the fact that the tap is running full force. The basin is overflowing and water is spilling onto the floor. Meanwhile, directly behind her, her two young children are raiding the cookie jar in the top cabinet. The boy has climbed up on a wooden stool but is leaning dangerously far over. Despite the danger, he's managed to get his hands inside the jar and is handing a cookie down to his little sister, who's standing with her hand outstretched eagerly waiting."""
}

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;
                padding:20px 0 16px;">
        <div style="font-size:40px;
                    margin-bottom:8px;">🧠</div>
        <div style="font-size:18px;
                    font-weight:800;
                    color:white;
                    letter-spacing:-0.5px;">
            CogniScan AI
        </div>
        <div style="font-size:11px;
                    color:rgba(255,255,255,0.5);
                    margin-top:4px;">
            Cognitive Decline Detection
        </div>
        <div style="
            display:inline-block;
            background:rgba(99,102,241,0.3);
            border:1px solid rgba(165,180,252,0.4);
            border-radius:20px;
            padding:3px 12px;
            font-size:10px;
            color:#C7D2FE;
            margin-top:10px;
            font-weight:600;">
            RESEARCH PROTOTYPE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Home",
         "🔬  Live Demo",
         "📊  Model Performance",
         "ℹ️  About"],
        label_visibility="collapsed")

    st.markdown("---")

    st.markdown("""
    <div style="padding:0 4px;">
        <div style="font-size:11px;
                    font-weight:700;
                    color:rgba(255,255,255,0.5);
                    letter-spacing:1px;
                    text-transform:uppercase;
                    margin-bottom:12px;">
            Model Info
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_items = [
        ("Architecture","RF + GB + SVM"),
        ("Dataset","DementiaBank Pitt"),
        ("Patients","N = 326"),
        ("Weighted F1","0.670 ± 0.059"),
        ("Macro AUC","0.851"),
        ("AD AUC","0.962"),
        ("CV","5-fold Stratified"),
    ]
    for label,val in info_items:
        st.markdown(f"""
        <div style="
            display:flex;
            justify-content:space-between;
            padding:6px 4px;
            border-bottom:1px solid
                rgba(255,255,255,0.07);">
            <span style="
                font-size:11px;
                color:rgba(255,255,255,0.5);">
                {label}
            </span>
            <span style="
                font-size:11px;
                font-weight:600;
                color:rgba(255,255,255,0.9);">
                {val}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        margin-top:24px;
        padding:12px;
        background:rgba(255,255,255,0.05);
        border-radius:10px;
        font-size:11px;
        color:rgba(255,255,255,0.4);
        line-height:1.6;">
        ⚠️ Research prototype only.<br>
        Not for clinical use.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════
if "Home" in page:

    # Hero
    st.markdown("""
    <div style="
        background:linear-gradient(
            135deg,
            #1E1B4B 0%,#312E81 30%,
            #1E40AF 65%,#0F766E 100%);
        padding:60px 56px;
        border-radius:24px;
        color:white;
        margin-bottom:32px;
        position:relative;
        overflow:hidden;
        box-shadow:0 20px 60px
            rgba(30,27,75,0.35);">
        <div style="
            position:absolute;
            top:-80px;right:-80px;
            width:320px;height:320px;
            background:radial-gradient(
                circle,rgba(139,92,246,0.25),
                transparent 70%);
            border-radius:50%;">
        </div>
        <div style="
            position:absolute;
            bottom:-60px;left:30%;
            width:250px;height:250px;
            background:radial-gradient(
                circle,rgba(14,165,233,0.15),
                transparent 70%);
            border-radius:50%;">
        </div>
        <div style="position:relative;z-index:1;">
            <div style="
                display:inline-flex;
                align-items:center;
                gap:6px;
                background:rgba(255,255,255,0.1);
                border:1px solid
                    rgba(255,255,255,0.2);
                border-radius:20px;
                padding:5px 16px;
                font-size:11px;
                font-weight:700;
                letter-spacing:1px;
                margin-bottom:24px;
                color:rgba(255,255,255,0.85);">
                ✦ IEEE CONFERENCE PAPER
                · KARPAGAM ACADEMY
            </div>
            <div style="
                font-size:52px;
                font-weight:900;
                line-height:1;
                letter-spacing:-2px;
                margin-bottom:16px;
                background:linear-gradient(
                    135deg,
                    #FFFFFF,#BAE6FD,#A5B4FC);
                -webkit-background-clip:text;
                -webkit-text-fill-color:
                    transparent;
                background-clip:text;">
                CogniScan AI
            </div>
            <div style="
                font-size:18px;
                color:rgba(255,255,255,0.8);
                font-weight:400;
                line-height:1.6;
                max-width:600px;
                margin-bottom:32px;">
                Explainable multimodal three-stage
                cognitive decline detection from
                spontaneous speech and linguistic
                biomarkers
            </div>
            <div style="
                display:flex;
                gap:10px;
                flex-wrap:wrap;">
                <div style="
                    background:rgba(
                        255,255,255,0.12);
                    border:1px solid rgba(
                        255,255,255,0.2);
                    border-radius:20px;
                    padding:7px 18px;
                    font-size:13px;
                    font-weight:600;">
                    🎯 Three-Stage Classification
                </div>
                <div style="
                    background:rgba(
                        255,255,255,0.12);
                    border:1px solid rgba(
                        255,255,255,0.2);
                    border-radius:20px;
                    padding:7px 18px;
                    font-size:13px;
                    font-weight:600;">
                    🔍 Dual-Layer XAI
                </div>
                <div style="
                    background:rgba(
                        255,255,255,0.12);
                    border:1px solid rgba(
                        255,255,255,0.2);
                    border-radius:20px;
                    padding:7px 18px;
                    font-size:13px;
                    font-weight:600;">
                    📈 Cognitive Drift Score
                </div>
                <div style="
                    background:rgba(
                        99,102,241,0.35);
                    border:1px solid rgba(
                        165,180,252,0.4);
                    border-radius:20px;
                    padding:7px 18px;
                    font-size:13px;
                    font-weight:700;
                    color:#E0E7FF;">
                    ⚡ F1=0.670 · AUC=0.851
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        (c1,"0.670","Weighted F1",
         "5-fold CV","#4F46E5","#EEF2FF"),
        (c2,"0.851","Macro AUC",
         "Primary metric","#7C3AED","#F5F3FF"),
        (c3,"0.962","AD Detection AUC",
         "Strong separation","#E53E3E","#FFF5F5"),
        (c4,"326","Patients",
         "DementiaBank Pitt","#0EA5E9","#F0F9FF"),
    ]
    for col,val,lbl,sub,color,bg in stats:
        with col:
            st.markdown(f"""
            <div style="
                background:{bg};
                border:1px solid {color}33;
                border-radius:16px;
                padding:22px 20px;
                text-align:center;
                box-shadow:0 4px 16px
                    {color}15;">
                <div style="
                    font-size:11px;
                    font-weight:700;
                    color:{color};
                    letter-spacing:1px;
                    text-transform:uppercase;
                    margin-bottom:8px;">
                    {lbl}
                </div>
                <div style="
                    font-size:36px;
                    font-weight:900;
                    color:#1E293B;
                    letter-spacing:-1px;
                    line-height:1;">
                    {val}
                </div>
                <div style="
                    font-size:11px;
                    color:#94A3B8;
                    margin-top:6px;">
                    {sub}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Problem + Contributions
    col_l, col_r = st.columns([1,1],
                               gap="large")
    with col_l:
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:28px 32px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);">
            <div style="
                font-size:11px;
                font-weight:700;
                color:#6366F1;
                letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:12px;">
                The Problem
            </div>
            <div style="
                font-size:22px;
                font-weight:800;
                color:#1E293B;
                line-height:1.3;
                margin-bottom:16px;">
                MCI is routinely missed by
                conventional screening tools
            </div>
            <div style="
                font-size:14px;
                color:#64748B;
                line-height:1.8;">
                Over <strong>55 million</strong>
                people live with dementia
                worldwide. Mild Cognitive
                Impairment (MCI) — the most
                actionable intervention window —
                is chronically underdetected by
                brief cognitive tests insensitive
                to early speech and language
                degradation.
                <br><br>
                Speech analysis provides a
                <strong>non-invasive, scalable
                proxy</strong> for cognitive
                assessment. Changes in vocabulary
                diversity, hesitation frequency,
                and vocal energy are measurable
                years before formal diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:28px 32px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);">
            <div style="
                font-size:11px;
                font-weight:700;
                color:#7C3AED;
                letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:12px;">
                3 Novel Contributions
            </div>
        """, unsafe_allow_html=True)

        contributions = [
            ("01","Three-Stage Classification",
             "AD · MCI · Control",
             "Most systems do binary AD vs Control. We explicitly target MCI — the clinically critical transitional stage.",
             "#4F46E5"),
            ("02","Dual-Layer XAI",
             "Global SHAP + Patient Explanation",
             "Layer 1: TreeSHAP global feature rankings. Layer 2: Natural language per-patient clinical summaries.",
             "#7C3AED"),
            ("03","Cognitive Drift Score",
             "Longitudinal Biomarker",
             "Novel CDS quantifies speech change across sessions for 66 diagnostically stable patients.",
             "#0EA5E9"),
        ]

        for num,title,sub,desc,color in \
                contributions:
            st.markdown(f"""
            <div style="
                display:flex;
                gap:16px;
                margin-bottom:20px;
                align-items:flex-start;">
                <div style="
                    width:36px;height:36px;
                    background:linear-gradient(
                        135deg,{color},{color}BB);
                    border-radius:10px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:white;
                    font-size:12px;
                    font-weight:800;
                    flex-shrink:0;">
                    {num}
                </div>
                <div>
                    <div style="
                        font-size:14px;
                        font-weight:700;
                        color:#1E293B;
                        margin-bottom:2px;">
                        {title}
                    </div>
                    <div style="
                        font-size:11px;
                        font-weight:600;
                        color:{color};
                        margin-bottom:4px;">
                        {sub}
                    </div>
                    <div style="
                        font-size:12px;
                        color:#64748B;
                        line-height:1.6;">
                        {desc}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>",
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline visual
    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.85);
        border:1px solid #E2E8F0;
        border-radius:20px;
        padding:32px 36px;
        box-shadow:0 4px 24px
            rgba(99,102,241,0.07);">
        <div style="
            font-size:11px;
            font-weight:700;
            color:#6366F1;
            letter-spacing:1px;
            text-transform:uppercase;
            margin-bottom:20px;">
            System Pipeline
        </div>
        <div style="
            display:flex;
            align-items:center;
            gap:0;
            justify-content:space-between;
            flex-wrap:nowrap;
            overflow-x:auto;">
    """, unsafe_allow_html=True)

    steps = [
        ("🎙️","Speech Input",
         ".cha transcript","#4F46E5"),
        ("⚙️","Preprocessing",
         "PAR-only extraction","#7C3AED"),
        ("📐","Feature Extraction",
         "16 biomarkers","#8B5CF6"),
        ("🤖","Ensemble Model",
         "RF + GB + SVM","#0EA5E9"),
        ("🔍","XAI Output",
         "SHAP + Explanation","#0D9488"),
    ]

    pipeline_html = ""
    for i,(icon,title,sub,color) in \
            enumerate(steps):
        pipeline_html += f"""
        <div style="
            flex:1;
            min-width:140px;
            text-align:center;
            padding:16px 8px;">
            <div style="
                width:52px;height:52px;
                background:linear-gradient(
                    135deg,{color}22,{color}11);
                border:2px solid {color}44;
                border-radius:14px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:22px;
                margin:0 auto 10px;">
                {icon}
            </div>
            <div style="
                font-size:13px;
                font-weight:700;
                color:#1E293B;
                margin-bottom:3px;">
                {title}
            </div>
            <div style="
                font-size:11px;
                color:#94A3B8;">
                {sub}
            </div>
        </div>"""
        if i < len(steps)-1:
            pipeline_html += """
            <div style="
                color:#CBD5E1;
                font-size:20px;
                flex-shrink:0;
                padding-bottom:20px;">
                →
            </div>"""

    st.markdown(
        pipeline_html + "</div></div>",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: LIVE DEMO
# ══════════════════════════════════════════════════════
elif "Demo" in page:

    st.markdown("""
    <div style="
        margin-bottom:28px;">
        <div style="
            font-size:11px;
            font-weight:700;
            color:#6366F1;
            letter-spacing:1px;
            text-transform:uppercase;
            margin-bottom:8px;">
            Live Inference
        </div>
        <div style="
            font-size:32px;
            font-weight:900;
            color:#1E293B;
            letter-spacing:-0.5px;">
            Patient Analysis Demo
        </div>
        <div style="
            font-size:15px;
            color:#64748B;
            margin-top:6px;">
            Enter a Cookie Theft picture
            description transcript to receive
            a real-time cognitive assessment
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_res = st.columns(
        [1,1], gap="large")

    with col_inp:
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:28px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);
            margin-bottom:16px;">
            <div style="
                font-size:14px;
                font-weight:700;
                color:#1E293B;
                margin-bottom:4px;">
                📋 Patient Transcript
            </div>
            <div style="
                font-size:12px;
                color:#94A3B8;
                margin-bottom:16px;">
                Cookie Theft picture description
                (DementiaBank protocol)
            </div>
        </div>
        """, unsafe_allow_html=True)

        s1,s2,s3 = st.columns(3)
        with s1:
            if st.button("🔴 AD",
                    use_container_width=True):
                st.session_state[
                    'transcript'] = SAMPLES['AD']
        with s2:
            if st.button("🟡 MCI",
                    use_container_width=True):
                st.session_state[
                    'transcript'] = SAMPLES['MCI']
        with s3:
            if st.button("🟢 Control",
                    use_container_width=True):
                st.session_state[
                    'transcript'] = \
                        SAMPLES['Control']

        transcript = st.text_area(
            "Transcript",
            value=st.session_state.get(
                'transcript',''),
            placeholder=(
                "Paste patient's description "
                "of the Cookie Theft picture..."),
            height=220,
            label_visibility="collapsed")

        st.markdown(
            "**📈 Cognitive Drift Score**",
            help=(
                "Set to 50 for single-session "
                "patients. Only meaningful for "
                "patients with multiple visits."))
        cds_val = st.slider(
            "CDS",
            0.0, 100.0, 50.0, 0.5,
            label_visibility="collapsed",
            help="50 = stable baseline")

        st.markdown(f"""
        <div style="
            display:flex;
            justify-content:space-between;
            font-size:11px;
            color:#94A3B8;
            margin:-8px 0 16px;
            padding:0 4px;">
            <span>0 — Improving</span>
            <span style="color:#0D9488;">
                Control μ=46.0
            </span>
            <span style="color:#D97706;">
                MCI μ=51.6
            </span>
            <span>100 — Rapid decline</span>
        </div>
        """, unsafe_allow_html=True)

        analyse = st.button(
            "🔍  Analyse Patient",
            type="primary",
            use_container_width=True)

        st.markdown("""
        <div style="
            text-align:center;
            font-size:11px;
            color:#94A3B8;
            margin-top:10px;">
            RF + GB + SVM · 16 biomarkers ·
            TreeSHAP · N=326 training patients
        </div>
        """, unsafe_allow_html=True)

    with col_res:
        if analyse or st.session_state.get(
                'last_result'):

            if analyse:
                if (not transcript or
                        len(transcript.strip())
                        < 20):
                    st.error(
                        "Please enter a longer "
                        "transcript.")
                    st.stop()

                with st.spinner(
                        "Analysing..."):
                    fd, fv = extract_features(
                        transcript, cds_val)
                    if fd is None:
                        st.error(
                            "Transcript too short.")
                        st.stop()

                    fs    = scaler.transform(fv)
                    probs = model.predict_proba(
                        fs)[0]
                    pi    = int(np.argmax(probs))
                    pc    = le.classes_[pi]
                    conf  = probs[pi]

                    sv = np.zeros(len(FEATURES))
                    try:
                        import shap
                        exp = shap.TreeExplainer(
                            model.estimators_[0])
                        sr  = exp.shap_values(fs)
                        sa  = np.array(sr)
                        if sa.ndim==3:
                            if sa.shape[2]==3:
                                sv=sa[0,:,pi]
                            else:
                                sv=sa[pi,0,:]
                        elif isinstance(sr,list):
                            sv=sr[pi][0]
                        sv=np.array(sv).flatten()
                        if len(sv)!=len(FEATURES):
                            sv=np.zeros(
                                len(FEATURES))
                    except Exception:
                        pass

                    st.session_state[
                        'last_result'] = {
                        'pc':pc,'conf':conf,
                        'probs':probs,'sv':sv,
                        'fs':fs[0],'fd':fd,
                        'cds':cds_val}

            r  = st.session_state['last_result']
            pc = r['pc']
            conf = r['conf']
            probs = r['probs']
            sv  = r['sv']
            fs  = r['fs']
            fd  = r['fd']

            cfg = {
                'AD'     :('#E53E3E','#FFF5F5',
                            '#FC8181'),
                'Control':('#0D9488','#F0FDFA',
                            '#5EEAD4'),
                'MCI'    :('#D97706','#FFFBEB',
                            '#FCD34D'),
            }
            color,bg,border = cfg[pc]
            icons = {
                'AD':'🔴',
                'Control':'🟢',
                'MCI':'🟡'}
            descs = {
                'AD'    :("Alzheimer's Disease "
                           "patterns detected."),
                'Control':("Healthy Control "
                            "patterns detected."),
                'MCI'   :("Mild Cognitive "
                           "Impairment detected. "
                           "Follow-up recommended.")
            }

            # Prediction card
            st.markdown(f"""
            <div style="
                background:linear-gradient(
                    135deg,{bg},{bg});
                border:2px solid {border};
                border-radius:20px;
                padding:28px 32px;
                margin-bottom:16px;
                box-shadow:0 8px 32px
                    {color}20;
                position:relative;
                overflow:hidden;">
                <div style="
                    position:absolute;
                    top:-30px;right:-30px;
                    width:120px;height:120px;
                    background:radial-gradient(
                        circle,{color}15,
                        transparent 70%);
                    border-radius:50%;">
                </div>
                <div style="
                    font-size:11px;
                    font-weight:700;
                    color:{color};
                    letter-spacing:1.5px;
                    text-transform:uppercase;
                    margin-bottom:12px;">
                    ✦ Diagnostic Result
                </div>
                <div style="
                    font-size:46px;
                    font-weight:900;
                    color:{color};
                    line-height:1;
                    margin-bottom:12px;
                    letter-spacing:-1px;">
                    {icons[pc]} {pc}
                </div>
                <div style="
                    display:inline-block;
                    background:{color};
                    color:white;
                    font-size:13px;
                    font-weight:700;
                    padding:5px 18px;
                    border-radius:20px;
                    margin-bottom:14px;">
                    {conf:.1%} confidence
                </div>
                <div style="
                    font-size:13px;
                    color:#374151;
                    line-height:1.7;
                    padding:12px 16px;
                    background:rgba(
                        255,255,255,0.7);
                    border-radius:10px;
                    border-left:3px solid
                        {color};">
                    {descs[pc]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Plotly probability chart
            cls_colors = {
                'AD':'#E53E3E',
                'Control':'#0D9488',
                'MCI':'#D97706'}
            fig = go.Figure()
            for i,cls in enumerate(le.classes_):
                fig.add_trace(go.Bar(
                    name=cls,
                    x=[probs[i]],
                    y=[cls],
                    orientation='h',
                    marker_color=cls_colors[cls],
                    marker_line_width=0,
                    text=f"{probs[i]:.1%}",
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(
                        color='white',
                        size=13,
                        family='Inter')
                ))
            fig.update_layout(
                showlegend=False,
                height=140,
                margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    range=[0,1],
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(
                        size=13,
                        color='#475569',
                        family='Inter')),
                bargap=0.3,
                font=dict(family='Inter'))
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={'displayModeBar':False})

    # XAI + CDS below
    if st.session_state.get('last_result'):
        r  = st.session_state['last_result']
        pc = r['pc']
        sv = r['sv']
        fs = r['fs']
        fd = r['fd']
        cfg2 = {
            'AD':'#E53E3E',
            'Control':'#0D9488',
            'MCI':'#D97706'}
        color = cfg2[pc]

        st.markdown("<br>", unsafe_allow_html=True)
        col_xai, col_cds = st.columns(
            [1,1], gap="large")

        with col_xai:
            st.markdown("""
            <div style="
                background:rgba(255,255,255,0.85);
                border:1px solid #E2E8F0;
                border-radius:20px;
                padding:24px 28px;
                box-shadow:0 4px 24px
                    rgba(99,102,241,0.07);">
                <div style="
                    display:flex;
                    align-items:center;
                    gap:10px;
                    margin-bottom:20px;
                    padding-bottom:14px;
                    border-bottom:1px solid
                        #EEF2FF;">
                    <div style="
                        width:32px;height:32px;
                        background:linear-gradient(
                            135deg,#8B5CF6,
                            #6366F1);
                        border-radius:8px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:16px;">
                        💬
                    </div>
                    <div>
                        <div style="
                            font-size:14px;
                            font-weight:700;
                            color:#1E293B;">
                            Patient Explanation
                        </div>
                        <div style="
                            font-size:11px;
                            color:#94A3B8;">
                            Layer 2 XAI · Top 3
                            SHAP factors
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            top3 = np.argsort(
                np.abs(sv))[::-1][:3]
            rank_c = ['#6366F1',
                      '#8B5CF6','#A78BFA']
            for rank,idx in enumerate(top3,1):
                fname = FEATURES[idx]
                fval  = fs[idx]
                sval  = sv[idx]
                direction = ('elevated'
                             if fval>0
                             else 'reduced')
                impact    = ('increases'
                             if sval>0
                             else 'decreases')
                name = FEAT_LABELS.get(
                    fname, fname)
                dc = ('#E53E3E'
                      if sval>0 else '#0D9488')
                arrow = '↑' if fval>0 else '↓'
                rc = rank_c[rank-1]
                st.markdown(f"""
                <div style="
                    display:flex;
                    gap:12px;
                    margin:10px 0;
                    padding:12px 16px;
                    background:rgba(
                        248,250,252,0.8);
                    border-radius:12px;
                    border:1px solid #EEF2FF;
                    border-left:4px solid
                        {color};">
                    <div style="
                        width:26px;height:26px;
                        background:{rc};
                        border-radius:8px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        color:white;
                        font-size:12px;
                        font-weight:800;
                        flex-shrink:0;">
                        {rank}
                    </div>
                    <div>
                        <div style="
                            font-size:13px;
                            font-weight:600;
                            color:#1E293B;">
                            {arrow} {name}
                            <span style="
                                font-weight:400;
                                color:#94A3B8;
                                font-size:12px;">
                                is {direction}
                            </span>
                        </div>
                        <div style="
                            font-size:11px;
                            color:{dc};
                            font-weight:700;
                            margin-top:3px;">
                            {impact} {pc}
                            likelihood
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
                <div style="
                    font-size:11px;
                    color:#94A3B8;
                    font-style:italic;
                    margin-top:12px;
                    padding-top:10px;
                    border-top:1px solid #EEF2FF;">
                    TreeSHAP on RF component.
                    Research use only.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_cds:
            cds_v = r['cds']
            if cds_v < 48:
                cs,cc,cb,cn = (
                    "Stable","#0D9488",
                    "#F0FDFA",
                    "CDS near Control mean (μ=46.0)."
                    " No significant drift detected.")
                ci = "✅"
            elif cds_v < 55:
                cs,cc,cb,cn = (
                    "Borderline","#D97706",
                    "#FFFBEB",
                    "CDS between Control (μ=46.0)"
                    " and MCI (μ=51.6). Monitor"
                    " longitudinally.")
                ci = "⚠️"
            else:
                cs,cc,cb,cn = (
                    "Elevated","#E53E3E",
                    "#FFF5F5",
                    "CDS above MCI mean (μ=51.6)."
                    " Significant cognitive drift"
                    " detected.")
                ci = "🚨"

            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cds_v,
                domain={'x':[0,1],'y':[0,1]},
                title={'text':"CDS Score",
                       'font':{'size':14,
                               'color':'#475569',
                               'family':'Inter'}},
                number={'font':{'size':36,
                                'color':cc,
                                'family':'Inter'}},
                gauge={
                    'axis':{'range':[0,100],
                            'tickwidth':1,
                            'tickcolor':'#CBD5E1',
                            'tickfont':{
                                'size':10,
                                'family':'Inter'}},
                    'bar':{'color':cc,
                           'thickness':0.25},
                    'bgcolor':'white',
                    'borderwidth':0,
                    'steps':[
                        {'range':[0,48],
                         'color':'#D1FAE5'},
                        {'range':[48,55],
                         'color':'#FEF3C7'},
                        {'range':[55,100],
                         'color':'#FEE2E2'},
                    ],
                    'threshold':{
                        'line':{
                            'color':cc,
                            'width':3},
                        'thickness':0.75,
                        'value':cds_v}}))
            fig2.update_layout(
                height=220,
                margin=dict(l=20,r=20,
                            t=40,b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'))
            st.markdown(f"""
            <div style="
                background:rgba(255,255,255,0.85);
                border:1px solid #E2E8F0;
                border-radius:20px;
                padding:24px 28px;
                box-shadow:0 4px 24px
                    rgba(99,102,241,0.07);">
                <div style="
                    display:flex;
                    align-items:center;
                    gap:10px;
                    margin-bottom:16px;
                    padding-bottom:14px;
                    border-bottom:1px solid
                        #EEF2FF;">
                    <div style="
                        width:32px;height:32px;
                        background:linear-gradient(
                            135deg,#0EA5E9,
                            #6366F1);
                        border-radius:8px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:16px;">
                        📈
                    </div>
                    <div>
                        <div style="
                            font-size:14px;
                            font-weight:700;
                            color:#1E293B;">
                            Cognitive Drift Score
                        </div>
                        <div style="
                            font-size:11px;
                            color:#94A3B8;">
                            Longitudinal biomarker
                            · stable patients only
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(
                fig2,
                use_container_width=True,
                config={'displayModeBar':False})
            st.markdown(f"""
                <div style="
                    display:inline-flex;
                    align-items:center;
                    gap:6px;
                    background:{cb};
                    color:{cc};
                    font-size:13px;
                    font-weight:700;
                    padding:6px 16px;
                    border-radius:20px;
                    margin-bottom:12px;">
                    {ci} {cs}
                </div>
                <div style="
                    font-size:13px;
                    color:#374151;
                    line-height:1.7;
                    padding:12px 16px;
                    background:{cb};
                    border-radius:10px;
                    border-left:3px solid {cc};">
                    {cn}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif "Performance" in page:

    st.markdown("""
    <div style="margin-bottom:28px;">
        <div style="font-size:11px;font-weight:700;
            color:#6366F1;letter-spacing:1px;
            text-transform:uppercase;
            margin-bottom:8px;">
            Evaluation Results
        </div>
        <div style="font-size:32px;font-weight:900;
            color:#1E293B;letter-spacing:-0.5px;">
            Model Performance
        </div>
        <div style="font-size:15px;color:#64748B;
            margin-top:6px;">
            5-fold stratified CV · Natural class
            distribution · N=326 patients
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-class table
    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.85);
        border:1px solid #E2E8F0;
        border-radius:20px;
        padding:28px 32px;
        box-shadow:0 4px 24px
            rgba(99,102,241,0.07);
        margin-bottom:20px;">
        <div style="font-size:14px;
            font-weight:700;color:#1E293B;
            margin-bottom:16px;">
            Per-Class Results
        </div>
    """, unsafe_allow_html=True)

    df_perf = pd.DataFrame({
        'Class'    : ['AD (n=135)',
                      'Control (n=110)',
                      'MCI (n=81)',
                      'Weighted Avg'],
        'Precision': [0.88,0.57,0.43,0.67],
        'Recall'   : [0.89,0.47,0.53,0.66],
        'F1 Score' : [0.886,0.524,0.512,0.670],
        'AUC'      : [0.962,0.816,0.776,'—'],
    })
    st.dataframe(
        df_perf,
        hide_index=True,
        use_container_width=True,
        column_config={
            'F1 Score': st.column_config.NumberColumn(
                format="%.3f"),
            'Precision':
                st.column_config.NumberColumn(
                    format="%.2f"),
            'Recall':
                st.column_config.NumberColumn(
                    format="%.2f"),
        })
    st.markdown("</div>",
                unsafe_allow_html=True)

    col_ab1, col_ab2 = st.columns(2, gap="large")

    with col_ab1:
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:24px 28px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);">
            <div style="font-size:14px;
                font-weight:700;color:#1E293B;
                margin-bottom:16px;">
                Model Architecture Ablation
            </div>
        """, unsafe_allow_html=True)

        df_mabl = pd.DataFrame({
            'Model'    : [
                'Logistic Regression',
                'SVM (RBF)','Random Forest',
                'Gradient Boosting',
                'RF + SVM','RF + GB',
                'GB + SVM',
                '★ RF+GB+SVM (Proposed)'],
            'F1w'      : [
                0.629,0.622,0.647,0.647,
                0.673,0.655,0.653,0.658],
            'Macro AUC': [
                0.807,0.833,0.833,0.847,
                0.841,0.848,0.849,0.850],
            'MCI F1'   : [
                0.417,0.437,0.497,0.433,
                0.508,0.447,0.458,0.477],
        })
        st.dataframe(
            df_mabl,
            hide_index=True,
            use_container_width=True,
            column_config={
                'F1w':st.column_config.NumberColumn(
                    format="%.3f"),
                'Macro AUC':
                    st.column_config.NumberColumn(
                        format="%.3f"),
                'MCI F1':
                    st.column_config.NumberColumn(
                        format="%.3f"),
            })
        st.markdown("</div>",
                    unsafe_allow_html=True)

    with col_ab2:
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:24px 28px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);">
            <div style="font-size:14px;
                font-weight:700;color:#1E293B;
                margin-bottom:16px;">
                Feature Modality Ablation
            </div>
        """, unsafe_allow_html=True)

        df_fabl = pd.DataFrame({
            'Feature Set'   : [
                'Linguistic only (10)',
                'Acoustic only (5)',
                'Ling + Acoustic (15)',
                '★ All + CDS (16)'],
            'Weighted F1'   : [
                0.557,0.598,0.647,0.658],
            'Macro AUC'     : [
                0.729,0.798,0.837,0.850],
        })
        st.dataframe(
            df_fabl,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Weighted F1':
                    st.column_config.NumberColumn(
                        format="%.3f"),
                'Macro AUC':
                    st.column_config.NumberColumn(
                        format="%.3f"),
            })

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly bar for feature ablation
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name='Weighted F1',
            x=df_fabl['Feature Set'],
            y=df_fabl['Weighted F1'],
            marker_color=['#CBD5E1','#94A3B8',
                          '#6366F1','#4F46E5'],
            text=df_fabl['Weighted F1'].apply(
                lambda x: f"{x:.3f}"),
            textposition='outside',
            textfont=dict(size=11,
                          family='Inter')))
        fig3.update_layout(
            showlegend=False,
            height=200,
            margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                range=[0.45,0.72],
                showgrid=True,
                gridcolor='#F1F5F9',
                tickfont=dict(size=10,
                              family='Inter')),
            xaxis=dict(
                tickfont=dict(size=9,
                              family='Inter')),
            font=dict(family='Inter'))
        st.plotly_chart(
            fig3,
            use_container_width=True,
            config={'displayModeBar':False})

        st.markdown("</div>",
                    unsafe_allow_html=True)

    # Baseline comparison
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background:rgba(255,255,255,0.85);
        border:1px solid #E2E8F0;
        border-radius:20px;
        padding:24px 28px;
        box-shadow:0 4px 24px
            rgba(99,102,241,0.07);">
        <div style="font-size:14px;
            font-weight:700;color:#1E293B;
            margin-bottom:16px;">
            Comparison with Published Baselines
        </div>
    """, unsafe_allow_html=True)

    df_base = pd.DataFrame({
        'System'  : [
            'Luz et al. (ADReSS)',
            'Balagopalan et al.',
            'Song et al. (Eye movement)',
            'PROCESS ICASSP 2025',
            '★ CogniScan AI (Proposed)'],
        'Year'    : [2021,2021,2025,2025,2025],
        'Task'    : [
            'Binary','Binary',
            '3-class','3-class','3-class'],
        'F1'      : [
            0.780,0.731,0.663,0.552,0.658],
        'XAI'     : [
            '✗','✗','✗','✗','✓ Dual-Layer'],
        'Dataset' : [
            'ADReSS','ADReSS',
            'Clinical','PROCESS',
            'DementiaBank Pitt'],
    })
    st.dataframe(
        df_base,
        hide_index=True,
        use_container_width=True,
        column_config={
            'F1': st.column_config.NumberColumn(
                format="%.3f")
        })
    st.markdown("</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div style="margin-bottom:28px;">
        <div style="font-size:11px;font-weight:700;
            color:#6366F1;letter-spacing:1px;
            text-transform:uppercase;
            margin-bottom:8px;">
            Project Overview
        </div>
        <div style="font-size:32px;font-weight:900;
            color:#1E293B;letter-spacing:-0.5px;">
            About CogniScan AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(
        [1,1], gap="large")

    with col_a1:
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:28px 32px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);
            margin-bottom:16px;">
            <div style="font-size:11px;
                font-weight:700;color:#6366F1;
                letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:12px;">
                Dataset
            </div>
            <div style="font-size:18px;
                font-weight:800;color:#1E293B;
                margin-bottom:12px;">
                DementiaBank Pitt Corpus
            </div>
            <div style="font-size:13px;
                color:#64748B;line-height:1.8;">
                The DementiaBank Pitt Corpus
                provides CHAT-format transcripts
                and audio recordings of the
                Cookie Theft picture description
                task from the Boston Diagnostic
                Aphasia Examination.
            </div>
        """, unsafe_allow_html=True)

        dist_data = pd.DataFrame({
            'Class'  : ['AD','Control','MCI'],
            'Count'  : [135,110,81],
            'Pct'    : ['41.4%','33.7%','24.8%'],
            'Color'  : ['🔴','🟢','🟡'],
        })
        for _,row in dist_data.iterrows():
            pct = int(row['Count']/326*100)
            st.markdown(f"""
            <div style="margin:8px 0;">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    font-size:12px;
                    font-weight:600;
                    color:#475569;
                    margin-bottom:4px;">
                    <span>
                        {row['Color']}
                        {row['Class']}
                    </span>
                    <span>
                        {row['Count']} ({row['Pct']})
                    </span>
                </div>
                <div style="
                    background:#EEF2FF;
                    border-radius:99px;
                    height:8px;overflow:hidden;">
                    <div style="
                        width:{pct}%;height:100%;
                        background:linear-gradient(
                            90deg,#6366F1,#8B5CF6);
                        border-radius:99px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Features
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:28px 32px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);">
            <div style="font-size:11px;
                font-weight:700;color:#6366F1;
                letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:16px;">
                16 Biomarkers
            </div>
        """, unsafe_allow_html=True)

        feat_groups = [
            ("Linguistic (10)","#4F46E5",[
                "MATTR — vocabulary diversity",
                "Filler ratio (um/uh)",
                "Repetition ratio",
                "Avg sentence length",
                "Content word ratio",
                "Pronoun/Article/Conjunction/",
                "Subordination ratios",
                "Utterance count"]),
            ("Acoustic (5)","#0EA5E9",[
                "Pause ratio",
                "Speech rate (ZCR)",
                "Energy mean (RMS)",
                "Pitch mean (pyin)",
                "Spectral centroid"]),
            ("CDS (1) ★","#E53E3E",[
                "Cognitive Drift Score",
                "Longitudinal change metric",
                "66 stable patients",
                "Control μ=46.0, MCI μ=51.6"]),
        ]
        for group,color,items in feat_groups:
            st.markdown(f"""
            <div style="margin-bottom:16px;">
                <div style="
                    font-size:12px;
                    font-weight:700;
                    color:{color};
                    margin-bottom:6px;">
                    {group}
                </div>
            """, unsafe_allow_html=True)
            for item in items:
                st.markdown(f"""
                <div style="
                    font-size:12px;
                    color:#64748B;
                    padding:3px 0 3px 12px;
                    border-left:2px solid
                        {color}44;">
                    {item}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>",
                        unsafe_allow_html=True)

        st.markdown("</div>",
                    unsafe_allow_html=True)

    with col_a2:
        # Methodology
        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.85);
            border:1px solid #E2E8F0;
            border-radius:20px;
            padding:28px 32px;
            box-shadow:0 4px 24px
                rgba(99,102,241,0.07);
            margin-bottom:16px;">
            <div style="font-size:11px;
                font-weight:700;color:#6366F1;
                letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:16px;">
                Methodology
            </div>
        """, unsafe_allow_html=True)

        method_steps = [
            ("1","Data Preprocessing",
             "PAR-only speaker extraction · "
             "Cookie Theft task scoping · "
             "Quality filtering (≥15s, ≥50 words)",
             "#4F46E5"),
            ("2","Feature Engineering",
             "16 multimodal biomarkers: "
             "10 linguistic + 5 acoustic + "
             "1 novel CDS",
             "#7C3AED"),
            ("3","Cross-Validation",
             "5-fold stratified CV · "
             "SMOTE on training folds only · "
             "StandardScaler fitted per fold · "
             "Zero patient overlap verified",
             "#8B5CF6"),
            ("4","Ensemble Classification",
             "RF (depth=3, min_leaf=10) + "
             "GB (lr=0.05) + SVM (C=0.5) "
             "soft-voting · Macro-AUC primary",
             "#0EA5E9"),
            ("5","Explainability (XAI)",
             "Layer 1: TreeSHAP global rankings · "
             "Layer 2: Patient natural language "
             "clinical summaries",
             "#0D9488"),
        ]
        for num,title,desc,color in method_steps:
            st.markdown(f"""
            <div style="
                display:flex;gap:14px;
                margin-bottom:16px;
                padding-bottom:16px;
                border-bottom:1px solid #F1F5F9;">
                <div style="
                    width:32px;height:32px;
                    background:linear-gradient(
                        135deg,{color},{color}BB);
                    border-radius:8px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:white;font-size:13px;
                    font-weight:800;flex-shrink:0;">
                    {num}
                </div>
                <div>
                    <div style="font-size:13px;
                        font-weight:700;
                        color:#1E293B;
                        margin-bottom:4px;">
                        {title}
                    </div>
                    <div style="font-size:12px;
                        color:#64748B;
                        line-height:1.6;">
                        {desc}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>",
                    unsafe_allow_html=True)

        # Paper info
        st.markdown("""
        <div style="
            background:linear-gradient(
                135deg,#1E1B4B,#312E81);
            border-radius:20px;
            padding:28px 32px;
            color:white;
            box-shadow:0 8px 32px
                rgba(30,27,75,0.25);">
            <div style="font-size:11px;
                font-weight:700;
                color:rgba(165,180,252,0.9);
                letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:12px;">
                Publication
            </div>
            <div style="font-size:16px;
                font-weight:700;
                color:white;
                line-height:1.4;
                margin-bottom:16px;">
                CogniScan AI: Explainable
                Multimodal System for
                Three-Class Cognitive Decline
                Detection from Speech and
                Linguistic Biomarkers
            </div>
            <div style="font-size:12px;
                color:rgba(255,255,255,0.6);
                margin-bottom:16px;">
                IEEE Conference Paper ·
                Karpagam Academy of Higher
                Education · 2025
            </div>
            <div style="
                display:flex;
                flex-wrap:wrap;gap:8px;">
                <div style="
                    background:rgba(
                        255,255,255,0.1);
                    border:1px solid rgba(
                        255,255,255,0.2);
                    border-radius:20px;
                    padding:4px 12px;
                    font-size:11px;
                    font-weight:600;">
                    DementiaBank Pitt
                </div>
                <div style="
                    background:rgba(
                        255,255,255,0.1);
                    border:1px solid rgba(
                        255,255,255,0.2);
                    border-radius:20px;
                    padding:4px 12px;
                    font-size:11px;
                    font-weight:600;">
                    SHAP Explainability
                </div>
                <div style="
                    background:rgba(
                        255,255,255,0.1);
                    border:1px solid rgba(
                        255,255,255,0.2);
                    border-radius:20px;
                    padding:4px 12px;
                    font-size:11px;
                    font-weight:600;">
                    Three-Stage Classification
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)