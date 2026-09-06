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
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero-banner {
    background: linear-gradient(135deg,
        #1B3A6B 0%, #2563A8 50%, #1D9E75 100%);
    padding: 36px 40px;
    border-radius: 16px;
    margin-bottom: 28px;
    color: white;
}
.hero-title {
    font-size: 32px;
    font-weight: 600;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 15px;
    opacity: 0.85;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    margin-top: 14px;
    font-weight: 500;
}

.metric-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-val {
    font-size: 26px;
    font-weight: 600;
    color: #1B3A6B;
    margin: 0;
}
.metric-lbl {
    font-size: 12px;
    color: #6B7280;
    margin: 4px 0 0;
    font-weight: 400;
}

.result-AD {
    background: #FEF2F2;
    border: 2px solid #C0392B;
    border-radius: 14px;
    padding: 24px 28px;
}
.result-Control {
    background: #F0FDF4;
    border: 2px solid #27AE60;
    border-radius: 14px;
    padding: 24px 28px;
}
.result-MCI {
    background: #FFFBEB;
    border: 2px solid #E67E22;
    border-radius: 14px;
    padding: 24px 28px;
}
.result-title {
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 6px;
}
.result-desc {
    font-size: 14px;
    color: #374151;
    margin: 0;
    line-height: 1.6;
}

.prob-bar-wrap {
    margin: 6px 0;
}
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 4px;
    color: #374151;
}
.prob-track {
    background: #F3F4F6;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

.explain-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563A8;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0;
}
.explain-box h4 {
    margin: 0 0 10px;
    font-size: 14px;
    color: #1B3A6B;
    font-weight: 600;
}
.explain-box ul {
    margin: 0;
    padding-left: 18px;
}
.explain-box li {
    font-size: 14px;
    color: #374151;
    margin-bottom: 5px;
    line-height: 1.5;
}
.explain-note {
    font-size: 11px;
    color: #9CA3AF;
    font-style: italic;
    margin-top: 10px;
}

.feat-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px;
}
.chip-ling { background:#EEF2FF; color:#3730A3; }
.chip-acou { background:#ECFDF5; color:#065F46; }
.chip-cds  { background:#FEF3C7; color:#92400E; }

.section-head {
    font-size: 16px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #F3F4F6;
}

.upload-hint {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 4px;
}

.sidebar-stat {
    background: #F8FAFC;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 6px 0;
}
.sidebar-stat-val {
    font-size: 18px;
    font-weight: 600;
    color: #1B3A6B;
}
.sidebar-stat-lbl {
    font-size: 11px;
    color: #6B7280;
}

.disclaimer {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #78350F;
    margin-top: 16px;
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
GLOBAL_MEDIAN = 50.0

# ── NLP constants ──────────────────────────────────────
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
FILLERS      = {'um','uh','mhm','hmm','er','ah',
                'well','okay'}
PRONOUNS     = {'i','me','my','mine','myself','he',
                'him','his','she','her','hers','it',
                'its','we','us','our','they','them',
                'their','you','your','this','that',
                'these','those'}
ARTICLES     = {'the','a','an'}
CONJUNCTIONS = {'and','but','or','nor','for','yet',
                'so','also','however','therefore',
                'although','though','because',
                'since','while','whereas'}
SUBORDINATORS= {'because','although','though',
                'while','when','after','before',
                'since','if','unless','until','as',
                'where','whether','that','which',
                'who','whom','whose'}

def compute_mattr(words, window=50):
    N = len(words)
    if N < window:
        return len(set(words))/N if N>0 else 0.0
    return float(np.mean([
        len(set(words[i:i+window]))/window
        for i in range(N-window+1)]))

def extract_par_text(cha_content):
    lines     = cha_content.split('\n')
    in_cookie = False
    par_lines = []
    has_gem   = any('@G:' in l for l in lines)
    for line in lines:
        line = line.strip()
        if line.startswith('@G:'):
            task = line.replace('@G:','').strip().lower()
            in_cookie = 'cookie' in task
            continue
        if has_gem and not in_cookie:
            continue
        if line.startswith('*PAR:'):
            text = line.replace('*PAR:','').strip()
            text = re.sub(r'\d+_\d+','',text)
            text = re.sub(r'\[.*?\]','',text)
            text = re.sub(r'&[+:=]\S+','',text)
            text = re.sub(r'[<>]','',text)
            text = re.sub(r'\(\.*\)','',text)
            text = re.sub(r'\+\.\.\.','',text)
            text = re.sub(r'xxx','',text)
            text = re.sub(r'\s+',' ',text).strip()
            if len(text)>2:
                par_lines.append(text)
    if not par_lines:
        for line in lines:
            line=line.strip()
            if line.startswith('*PAR:'):
                text=re.sub(r'\d+_\d+','',
                    line.replace('*PAR:','').strip())
                text=re.sub(r'\s+',' ',text).strip()
                if len(text)>2:
                    par_lines.append(text)
    return ' '.join(par_lines)

def extract_linguistic(transcript):
    text  = transcript.lower().strip()
    sents = [s.strip() for s in
             re.split(r'[.!?]',text)
             if len(s.strip())>2]
    words = re.findall(r"\b[a-z']+\b",text)
    N = len(words)
    if N < 5:
        return None
    V  = len(set(words))
    wf = Counter(words)
    return {
        'mattr'           : round(
            compute_mattr(words),4),
        'filler_ratio'    : round(
            sum(wf.get(f,0) for f in FILLERS)/N,4),
        'repetition_ratio': round(
            sum(1 for w,c in wf.items()
                if c>2)/V,4) if V>0 else 0.0,
        'avg_sent_len'    : round(float(np.mean(
            [len(s.split()) for s in sents])),4)
            if sents else 0.0,
        'content_ratio'   : round(
            len([w for w in words
                 if w not in STOPWORDS])/N,4),
        'pronoun_ratio'   : round(
            sum(wf.get(p,0)
                for p in PRONOUNS)/N,4),
        'article_ratio'   : round(
            sum(wf.get(a,0)
                for a in ARTICLES)/N,4),
        'conj_ratio'      : round(
            sum(wf.get(c,0)
                for c in CONJUNCTIONS)/N,4),
        'sub_ratio'       : round(
            sum(wf.get(s,0)
                for s in SUBORDINATORS)/N,4),
        'utterance_count' : len(sents),
    }

def extract_acoustic(audio_bytes):
    try:
        import librosa
        import tempfile
        with tempfile.NamedTemporaryFile(
                suffix='.mp3',delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        y,sr = librosa.load(
            tmp_path,sr=16000,mono=True)
        os.unlink(tmp_path)
        zcr  = librosa.feature.zero_crossing_rate(y)
        rms  = librosa.feature.rms(y=y)
        cent = librosa.feature.spectral_centroid(
            y=y,sr=sr)
        try:
            f0,_,_=librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr)
            pitch = float(np.nanmean(f0)) \
                    if f0 is not None and \
                    not np.all(np.isnan(f0)) \
                    else 0.0
        except Exception:
            pitch=0.0
        ev = rms[0]
        e_min=np.percentile(ev,5)
        e_max=np.percentile(ev,95)
        en   =np.clip((ev-e_min)/(e_max-e_min+1e-8),0,1)
        return {
            'speech_rate'  : float(np.mean(zcr)),
            'energy_mean'  : float(np.mean(rms)),
            'pitch_mean'   : pitch,
            'centroid_mean': float(np.mean(cent)),
            'pause_ratio'  : float(np.mean(1-en)),
        }
    except Exception as e:
        return None

FEAT_LABELS = {
    'mattr'           : 'Vocabulary diversity',
    'filler_ratio'    : 'Filler word ratio',
    'repetition_ratio': 'Word repetition',
    'avg_sent_len'    : 'Avg sentence length',
    'content_ratio'   : 'Content word ratio',
    'pronoun_ratio'   : 'Pronoun ratio',
    'article_ratio'   : 'Article ratio',
    'conj_ratio'      : 'Conjunction ratio',
    'sub_ratio'       : 'Subordination ratio',
    'utterance_count' : 'Utterance count',
    'pause_ratio'     : 'Pause ratio',
    'speech_rate'     : 'Speech rate',
    'energy_mean'     : 'Vocal energy',
    'pitch_mean'      : 'Voice pitch',
    'centroid_mean'   : 'Spectral centroid',
    'cds'             : 'Cognitive Drift Score',
}

def feat_type(f):
    if f=='cds': return 'cds'
    if f in ['pause_ratio','speech_rate',
             'energy_mean','pitch_mean',
             'centroid_mean']: return 'acou'
    return 'ling'

CLS_COLOR = {
    'AD':'#C0392B','Control':'#27AE60','MCI':'#E67E22'}
CLS_ICON  = {
    'AD':'🔴','Control':'🟢','MCI':'🟡'}
CLS_BG    = {
    'AD':'#FEF2F2','Control':'#F0FDF4','MCI':'#FFFBEB'}
CLS_DESC  = {
    'AD'     : ('Alzheimer\'s Disease detected. '
                'Speech patterns show significant '
                'acoustic and linguistic decline '
                'consistent with established AD. '
                'Immediate clinical evaluation '
                'is recommended.'),
    'Control': ('Healthy Control. Speech patterns '
                'are consistent with normal '
                'cognitive aging. Continued '
                'routine monitoring advised.'),
    'MCI'    : ('Mild Cognitive Impairment detected. '
                'Speech shows early-stage decline '
                'patterns at the Control-MCI '
                'boundary. Clinical follow-up '
                'and longitudinal monitoring '
                'strongly recommended.'),
}

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px">
        <div style="font-size:48px">🧠</div>
        <div style="font-size:18px;font-weight:600;
                    color:#1B3A6B">CogniScan AI</div>
        <div style="font-size:12px;color:#6B7280;
                    margin-top:4px">
            Cognitive Decline Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Model Performance")
    stats = [
        ("Weighted F1","0.658"),
        ("Macro AUC","0.850"),
        ("AD Detection AUC","0.962"),
        ("Dataset","N = 326"),
    ]
    for lbl,val in stats:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-val">{val}</div>
            <div class="sidebar-stat-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Diagnostic Classes")
    for cls,icon,col in [
            ('AD','🔴','#C0392B'),
            ('MCI','🟡','#E67E22'),
            ('Control','🟢','#27AE60')]:
        st.markdown(
            f"<span style='color:{col};"
            f"font-weight:600'>{icon} {cls}</span>",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Instructions")
    st.markdown(
        "1. Upload `.cha` transcript\n"
        "2. Upload `.mp3` audio *(optional)*\n"
        "3. Set CDS if longitudinal data available\n"
        "4. Click **Analyse Patient**")

    st.markdown("""
    <div class="disclaimer">
        ⚠️ For research use only. Not a substitute
        for clinical diagnosis.
    </div>""", unsafe_allow_html=True)

# ── Hero Banner ────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🧠 CogniScan AI</p>
    <p class="hero-subtitle">
        Explainable multimodal system for
        three-stage cognitive decline detection
        from spontaneous speech
    </p>
    <span class="hero-badge">
        RF + GB + SVM · DementiaBank Pitt ·
        Macro AUC 0.850
    </span>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ─────────────────────────────────────
st.markdown(
    '<p class="section-head">📂 Upload Patient Data</p>',
    unsafe_allow_html=True)

col1,col2,col3 = st.columns([2,2,1])

with col1:
    cha_file = st.file_uploader(
        "Transcript (.cha)",
        type=['cha'],
        label_visibility="visible")
    st.markdown(
        '<p class="upload-hint">DementiaBank '
        'CHAT format · Cookie Theft task</p>',
        unsafe_allow_html=True)

with col2:
    mp3_file = st.file_uploader(
        "Audio (.mp3 / .wav) — optional",
        type=['mp3','wav'],
        label_visibility="visible")
    st.markdown(
        '<p class="upload-hint">For acoustic '
        'feature extraction</p>',
        unsafe_allow_html=True)

with col3:
    st.markdown("**CDS Score**")
    cds_val = st.number_input(
        "CDS",
        min_value=0.0, max_value=100.0,
        value=50.0, step=0.5,
        label_visibility="collapsed")
    st.markdown(
        '<p class="upload-hint">50 = stable '
        'baseline · higher = more drift</p>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
analyse = st.button(
    "🔍 Analyse Patient",
    type="primary",
    use_container_width=True)

# ── Analysis ───────────────────────────────────────────
if analyse:
    if cha_file is None:
        st.error(
            "⚠️ Please upload a .cha "
            "transcript file to continue.")
        st.stop()

    with st.spinner(
            "Extracting features and "
            "computing prediction…"):

        cha_content = cha_file.read().decode(
            'utf-8', errors='ignore')
        transcript  = extract_par_text(cha_content)

        if len(transcript.split()) < 10:
            st.error(
                "Transcript too short. Ensure "
                "the file contains PAR speaker "
                "turns in the Cookie Theft task.")
            st.stop()

        ling = extract_linguistic(transcript)
        if ling is None:
            st.error(
                "Could not extract features. "
                "Check transcript format.")
            st.stop()

        acou = None
        audio_used = False
        if mp3_file is not None:
            acou = extract_acoustic(mp3_file.read())
            if acou:
                audio_used = True

        if acou is None:
            acou = {
                'pause_ratio'  : 0.50,
                'speech_rate'  : 0.12,
                'energy_mean'  : 0.045,
                'pitch_mean'   : 170.0,
                'centroid_mean': 1500.0,
            }

        feat_dict   = {**ling, **acou,
                       'cds': cds_val}
        feat_vec    = np.array(
            [feat_dict.get(f,0.0)
             for f in FEATURES]).reshape(1,-1)
        feat_scaled = scaler.transform(feat_vec)

        probs      = model.predict_proba(
            feat_scaled)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = le.classes_[pred_idx]
        confidence = probs[pred_idx]

        shap_vals  = np.zeros(len(FEATURES))
        shap_ok    = False
        try:
            import shap
            rf_comp   = model.estimators_[0]
            explainer = shap.TreeExplainer(rf_comp)
            sv_raw    = explainer.shap_values(
                feat_scaled)
            sv_arr    = np.array(sv_raw)
            if sv_arr.ndim==3:
                if sv_arr.shape[2]==3:
                    shap_vals=sv_arr[0,:,pred_idx]
                else:
                    shap_vals=sv_arr[pred_idx,0,:]
            else:
                shap_vals=sv_raw[pred_idx][0]
            shap_ok = True
        except Exception:
            pass

    # ── Results layout ────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p class="section-head">'
        '📊 Prediction Result</p>',
        unsafe_allow_html=True)

    # Main result card
    st.markdown(f"""
    <div class="result-{pred_class}">
        <p class="result-title" style="
            color:{CLS_COLOR[pred_class]}">
            {CLS_ICON[pred_class]} {pred_class}
            &nbsp;&nbsp;
            <span style="font-size:15px;
                         font-weight:400;
                         color:#6B7280">
                {confidence:.1%} confidence
            </span>
        </p>
        <p class="result-desc">
            {CLS_DESC[pred_class]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bars
    col_p,col_m = st.columns([3,1])
    with col_p:
        st.markdown("**Class probabilities**")
        order = np.argsort(probs)[::-1]
        for i in order:
            cls  = le.classes_[i]
            prob = probs[i]
            pct  = int(prob*100)
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-label">
                    <span>
                        {CLS_ICON[cls]} {cls}
                    </span>
                    <span style="color:
                        {CLS_COLOR[cls]};
                        font-weight:600">
                        {prob:.1%}
                    </span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="
                        width:{pct}%;
                        background:
                        {CLS_COLOR[cls]}">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_m:
        st.markdown("**Model info**")
        info_items = [
            ("Features","16"),
            ("Folds","5-fold CV"),
            ("F1w","0.658"),
            ("MacAUC","0.850"),
            ("Audio",
             "✅ Yes" if audio_used
             else "⚠️ Default"),
        ]
        for lbl,val in info_items:
            st.markdown(
                f"<div style='font-size:12px;"
                f"color:#6B7280'>{lbl}</div>"
                f"<div style='font-size:14px;"
                f"font-weight:600;color:#111827;"
                f"margin-bottom:8px'>{val}</div>",
                unsafe_allow_html=True)

    # ── SHAP Layer 1 ──────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p class="section-head">'
        '🔍 Layer 1 — Feature Importance (SHAP)</p>',
        unsafe_allow_html=True)

    if shap_ok:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        feat_order = np.argsort(
            np.abs(shap_vals))[::-1]

        fig,ax = plt.subplots(figsize=(9,5.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        bar_colors = []
        for i in feat_order:
            ft = feat_type(FEATURES[i])
            if ft=='cds':
                bar_colors.append('#C0392B')
            elif ft=='acou':
                bar_colors.append('#1D9E75')
            else:
                bar_colors.append('#2563A8')

        vals = shap_vals[feat_order]
        bars = ax.barh(
            range(len(FEATURES)), vals,
            color=bar_colors,
            alpha=0.85, edgecolor='white',
            height=0.65)

        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels(
            [FEAT_LABELS.get(FEATURES[i],FEATURES[i])
             for i in feat_order],
            fontsize=8.5)
        ax.axvline(0,'black',lw=0.8,
                   alpha=0.35,ls='--')
        ax.set_xlabel(
            f'SHAP value  '
            f'(positive = pushes toward '
            f'{pred_class})',
            fontsize=9)
        ax.set_title(
            f'Feature attributions — '
            f'{pred_class} prediction  '
            f'(TreeSHAP on RF component)',
            fontsize=10, fontweight='600',
            pad=12)
        patches = [
            mpatches.Patch(
                color='#2563A8',
                label='Linguistic (10)'),
            mpatches.Patch(
                color='#1D9E75',
                label='Acoustic (5)'),
            mpatches.Patch(
                color='#C0392B',
                label='CDS (1)'),
        ]
        ax.legend(handles=patches,fontsize=8,
                  loc='lower right',
                  framealpha=0.9)
        ax.spines[['top','right',
                   'left']].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis='x',alpha=0.15,lw=0.5)
        plt.tight_layout()
        st.pyplot(fig,use_container_width=True)
        plt.close()

        # Feature chips
        top3 = feat_order[:3]
        st.markdown("**Top contributing features:**")
        chips_html = ""
        for i in top3:
            ft   = feat_type(FEATURES[i])
            cls_ = f"chip_{ft}"
            name = FEAT_LABELS.get(
                FEATURES[i],FEATURES[i])
            val  = round(feat_dict.get(
                FEATURES[i],0.0),3)
            chips_html += (
                f'<span class="feat-chip {cls_}">'
                f'{name}: {val}</span>')
        st.markdown(chips_html,
                    unsafe_allow_html=True)
    else:
        st.info(
            "Install shap for feature "
            "importance: `pip install shap`")

    # ── Layer 2 ───────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p class="section-head">'
        '💬 Layer 2 — Clinical Explanation</p>',
        unsafe_allow_html=True)

    feat_name_map = {
        'mattr'           : 'vocabulary diversity',
        'filler_ratio'    : 'filler word usage',
        'repetition_ratio': 'word repetition',
        'avg_sent_len'    : 'sentence length',
        'content_ratio'   : 'content word usage',
        'pronoun_ratio'   : 'pronoun usage',
        'article_ratio'   : 'article usage',
        'conj_ratio'      : 'conjunction usage',
        'sub_ratio'       : 'sentence complexity',
        'utterance_count' : 'number of utterances',
        'pause_ratio'     : 'pausing during speech',
        'speech_rate'     : 'speech rate',
        'energy_mean'     : 'vocal energy',
        'pitch_mean'      : 'voice pitch',
        'centroid_mean'   : 'voice brightness',
        'cds'             : 'cognitive drift score',
    }

    top_idx = np.argsort(
        np.abs(shap_vals))[::-1][:3]
    reasons = []
    for idx in top_idx:
        fname = FEATURES[idx]
        fval  = feat_scaled[0,idx]
        sval  = shap_vals[idx]
        direction = ('elevated'
                     if fval>0 else 'reduced')
        impact    = ('increases'
                     if sval>0 else 'decreases')
        name = feat_name_map.get(fname,fname)
        reasons.append(
            f"{name.capitalize()} is "
            f"<strong>{direction}</strong> — "
            f"this {impact} "
            f"{pred_class} likelihood")

    reasons_li = "".join(
        f"<li>{r}</li>" for r in reasons)

    st.markdown(f"""
    <div class="explain-box" style="
        border-left-color:{CLS_COLOR[pred_class]}">
        <h4 style="color:{CLS_COLOR[pred_class]}">
            {CLS_ICON[pred_class]}
            Patient prediction: {pred_class}
            &nbsp;·&nbsp;
            {confidence:.1%} confidence
        </h4>
        <ul>{reasons_li}</ul>
        <p class="explain-note">
            Generated by TreeSHAP on the Random
            Forest component of the soft-voting
            ensemble. This explanation identifies
            the three most influential speech
            features for this patient's prediction.
            For research and clinical support only
            — not a substitute for clinical
            assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Transcript Preview ────────────────────────────
    st.markdown("---")
    with st.expander(
            "📄 Extracted participant transcript "
            "(PAR-only · Cookie Theft task)"):
        word_count = len(transcript.split())
        st.markdown(
            f"*{word_count} words extracted "
            f"from participant turns only*")
        st.markdown(f"> {transcript}")

    # ── Feature table ─────────────────────────────────
    with st.expander("📋 Full feature values"):
        rows = []
        for f in FEATURES:
            ft = feat_type(f)
            modality = (
                'CDS' if ft=='cds'
                else 'Acoustic' if ft=='acou'
                else 'Linguistic')
            rows.append({
                'Feature' : FEAT_LABELS.get(f,f),
                'Modality': modality,
                'Value'   : round(
                    feat_dict.get(f,0.0),4),
            })
        df_feat = pd.DataFrame(rows)
        st.dataframe(
            df_feat,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Value': st.column_config.NumberColumn(
                    format="%.4f")
            })