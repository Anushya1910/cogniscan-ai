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

# ── Load model artifacts ───────────────────────────────
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

# ── Feature extraction ─────────────────────────────────
STOPWORDS = {
    'the','a','an','and','or','but','in','on',
    'at','to','for','of','with','by','from',
    'is','are','was','were','be','been','has',
    'have','had','do','does','did','will','would',
    'could','should','may','might','shall','can',
    'not','no','nor','so','yet','both','either',
    'neither','each','few','more','most','other',
    'some','such','than','too','very','just',
    'as','if','then','that','this','these','those'
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
                'since','if','unless','until',
                'as','where','whether','that',
                'which','who','whom','whose'}

def compute_mattr(words, window=50):
    N = len(words)
    if N < window:
        return len(set(words))/N if N>0 else 0.0
    return float(np.mean([
        len(set(words[i:i+window]))/window
        for i in range(N-window+1)
    ]))

def extract_par_text(cha_content):
    """Extract PAR-only Cookie Theft utterances."""
    lines = cha_content.split('\n')
    in_cookie = False
    par_lines  = []
    has_gem    = any('@G:' in l for l in lines)

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
            if len(text) > 2:
                par_lines.append(text)

    if not par_lines and has_gem:
        in_cookie = False
        for line in lines:
            line = line.strip()
            if line.startswith('@G:'):
                in_cookie = True
                continue
            if in_cookie and line.startswith('*PAR:'):
                text = re.sub(r'\d+_\d+','',
                    line.replace('*PAR:','').strip())
                text = re.sub(r'\s+',' ',text).strip()
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
        'mattr'           : round(compute_mattr(words),4),
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

def extract_acoustic_from_audio(audio_bytes):
    """Extract 5 acoustic features using librosa."""
    try:
        import librosa
        import tempfile
        with tempfile.NamedTemporaryFile(
                suffix='.mp3',delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        y,sr = librosa.load(tmp_path,sr=16000,mono=True)
        os.unlink(tmp_path)

        zcr   = librosa.feature.zero_crossing_rate(y)
        rms   = librosa.feature.rms(y=y)
        cent  = librosa.feature.spectral_centroid(
            y=y,sr=sr)
        try:
            f0,_,_ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr)
            pitch = float(np.nanmean(f0)) \
                    if f0 is not None and \
                    not np.all(np.isnan(f0)) \
                    else 0.0
        except Exception:
            pitch = 0.0

        energy_vals = rms[0]
        e_min = np.percentile(energy_vals,5)
        e_max = np.percentile(energy_vals,95)
        energy_norm = np.clip(
            (energy_vals-e_min)/
            (e_max-e_min+1e-8),0,1)
        pause_ratio = float(np.mean(1-energy_norm))

        return {
            'speech_rate'  : float(np.mean(zcr)),
            'energy_mean'  : float(np.mean(rms)),
            'pitch_mean'   : pitch,
            'centroid_mean': float(np.mean(cent)),
            'pause_ratio'  : pause_ratio,
        }
    except Exception as e:
        st.warning(f"Audio processing note: {e}. "
                   f"Using text features only.")
        return None

def generate_explanation(feat_vals, shap_vals,
                          pred_class, confidence):
    """Generate natural language explanation."""
    feat_names = {
        'mattr'           : 'vocabulary diversity',
        'filler_ratio'    : 'filler word usage (um/uh)',
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
        fval  = feat_vals[idx]
        sval  = shap_vals[idx]
        direction = 'elevated' if fval>0 else 'reduced'
        impact    = 'increases' if sval>0 else 'decreases'
        name = feat_names.get(fname, fname)
        reasons.append(
            f"{name} is **{direction}** "
            f"({impact} {pred_class} likelihood)")
    return reasons

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/"
             "96/brain.png", width=64)
    st.title("CogniScan AI")
    st.caption(
        "Explainable three-stage cognitive "
        "decline detection from speech")
    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        "1. Upload a `.cha` transcript file\n"
        "2. Optionally upload the `.mp3` audio\n"
        "3. Click **Analyse** to get prediction\n"
        "4. Review SHAP explanation below")
    st.divider()
    st.markdown("**Classes**")
    st.markdown(
        "🔴 **AD** — Alzheimer's Disease\n\n"
        "🟡 **MCI** — Mild Cognitive Impairment\n\n"
        "🟢 **Control** — Healthy Control")
    st.divider()
    st.caption(
        "Model: RF+GB+SVM soft voting | "
        "Weighted F1=0.658 | Macro AUC=0.850 | "
        "DementiaBank Pitt Corpus N=326")

# ── Main UI ────────────────────────────────────────────
st.title("🧠 CogniScan AI")
st.subheader(
    "Explainable Multimodal Cognitive "
    "Decline Detection")
st.markdown(
    "Upload a Cookie Theft picture description "
    "transcript~(.cha) to receive an "
    "explainable three-stage cognitive "
    "assessment.")

col1,col2 = st.columns(2)
with col1:
    cha_file = st.file_uploader(
        "Upload transcript (.cha)",
        type=['cha'],
        help="DementiaBank CHAT format transcript")
with col2:
    mp3_file = st.file_uploader(
        "Upload audio (.mp3) — optional",
        type=['mp3','wav'],
        help="Participant audio for acoustic features")

cds_val = st.number_input(
    "Cognitive Drift Score (leave at 50.0 "
    "if no longitudinal data)",
    min_value=0.0, max_value=100.0,
    value=50.0, step=0.1,
    help="CDS=50 means stable. "
         "Higher values indicate more drift.")

analyse = st.button(
    "🔍 Analyse Patient",
    type="primary",
    use_container_width=True)

# ── Analysis ───────────────────────────────────────────
if analyse:
    if cha_file is None:
        st.error(
            "Please upload a .cha transcript file.")
        st.stop()

    with st.spinner("Extracting features..."):

        # Parse transcript
        cha_content = cha_file.read().decode(
            'utf-8', errors='ignore')
        transcript  = extract_par_text(cha_content)

        if len(transcript.split()) < 10:
            st.error(
                "Transcript too short to analyse. "
                "Ensure the file contains PAR "
                "speaker turns in Cookie Theft task.")
            st.stop()

        # Linguistic features
        ling = extract_linguistic(transcript)
        if ling is None:
            st.error(
                "Could not extract linguistic "
                "features. Check transcript format.")
            st.stop()

        # Acoustic features
        acou = None
        if mp3_file is not None:
            acou = extract_acoustic_from_audio(
                mp3_file.read())

        # Use defaults if no audio
        if acou is None:
            acou = {
                'pause_ratio'  : 0.50,
                'speech_rate'  : 0.12,
                'energy_mean'  : 0.045,
                'pitch_mean'   : 170.0,
                'centroid_mean': 1500.0,
            }
            if mp3_file is None:
                st.info(
                    "No audio uploaded. Acoustic "
                    "features set to population "
                    "medians. Upload audio for "
                    "personalised acoustic analysis.")

        # Build feature vector
        feat_dict = {**ling, **acou, 'cds': cds_val}
        feat_vec  = np.array(
            [feat_dict.get(f, 0.0)
             for f in FEATURES]).reshape(1,-1)
        feat_scaled = scaler.transform(feat_vec)

        # Predict
        probs      = model.predict_proba(feat_scaled)[0]
        pred_idx   = np.argmax(probs)
        pred_class = le.classes_[pred_idx]
        confidence = probs[pred_idx]

        # SHAP
        try:
            import shap
            rf_comp  = model.estimators_[0]
            explainer= shap.TreeExplainer(rf_comp)
            sv_raw   = explainer.shap_values(
                feat_scaled)
            sv_arr   = np.array(sv_raw)
            if sv_arr.ndim==3:
                if sv_arr.shape[2]==3:
                    shap_vals = sv_arr[0,:,pred_idx]
                else:
                    shap_vals = sv_arr[pred_idx,0,:]
            else:
                shap_vals = sv_raw[pred_idx][0]
            shap_ok = True
        except Exception:
            shap_vals = np.zeros(len(FEATURES))
            shap_ok   = False

    # ── Results ───────────────────────────────────────
    st.divider()
    st.subheader("📊 Prediction Result")

    cls_colors = {
        'AD'     : '#C0392B',
        'Control': '#27AE60',
        'MCI'    : '#E67E22'
    }
    cls_icons = {
        'AD'     : '🔴',
        'Control': '🟢',
        'MCI'    : '🟡'
    }
    cls_desc = {
        'AD'     : 'Alzheimer\'s Disease detected. '
                   'Speech patterns show significant '
                   'acoustic and linguistic decline '
                   'consistent with AD.',
        'Control': 'Healthy Control. Speech patterns '
                   'are consistent with normal '
                   'cognitive aging.',
        'MCI'    : 'Mild Cognitive Impairment detected. '
                   'Speech shows early-stage decline '
                   'patterns. Clinical follow-up '
                   'recommended.'
    }

    c1,c2,c3 = st.columns([2,1,1])
    with c1:
        st.markdown(
            f"### {cls_icons[pred_class]} "
            f"{pred_class}")
        st.markdown(cls_desc[pred_class])
    with c2:
        st.metric("Confidence",
                  f"{confidence:.1%}")
    with c3:
        st.metric("Macro AUC",
                  "0.850",
                  help="Population-level model AUC")

    # Probability bars
    st.markdown("**Class probabilities**")
    prob_df = pd.DataFrame({
        'Class'      : le.classes_,
        'Probability': probs
    }).sort_values('Probability',ascending=False)

    for _,row in prob_df.iterrows():
        col_a,col_b = st.columns([3,1])
        with col_a:
            st.progress(float(row['Probability']),
                        text=row['Class'])
        with col_b:
            st.write(f"{row['Probability']:.1%}")

    # ── Feature values ─────────────────────────────────
    st.divider()
    st.subheader("📋 Extracted Features")

    feat_display = pd.DataFrame({
        'Feature': FEATURES,
        'Value'  : [round(feat_dict.get(f,0.0),4)
                    for f in FEATURES]
    })

    col_l,col_r = st.columns(2)
    with col_l:
        st.markdown("**Linguistic**")
        st.dataframe(
            feat_display.iloc[:10],
            hide_index=True,
            use_container_width=True)
    with col_r:
        st.markdown("**Acoustic + CDS**")
        st.dataframe(
            feat_display.iloc[10:],
            hide_index=True,
            use_container_width=True)

    # ── SHAP Layer 1 ───────────────────────────────────
    st.divider()
    st.subheader("🔍 Layer 1 — Feature Importance "
                 "(SHAP)")

    if shap_ok:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        feat_order = np.argsort(
            np.abs(shap_vals))[::-1]

        fig,ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('white')

        colors_bar = [
            '#C0392B' if FEATURES[i]=='cds'
            else '#27AE60'
            if FEATURES[i] in [
                'pause_ratio','speech_rate',
                'energy_mean','pitch_mean',
                'centroid_mean']
            else '#2563A8'
            for i in feat_order
        ]
        vals_ordered = shap_vals[feat_order]

        bars = ax.barh(
            range(len(FEATURES)),
            vals_ordered,
            color=colors_bar,
            alpha=0.85,
            edgecolor='white')

        ax.set_yticks(range(len(FEATURES)))
        ax.set_yticklabels(
            [FEATURES[i] for i in feat_order],
            fontsize=8)
        ax.axvline(0,color='black',
                   lw=0.8,alpha=0.4,ls='--')
        ax.set_xlabel(
            f'SHAP value (impact on '
            f'{pred_class} prediction)',
            fontsize=9)
        ax.set_title(
            f'Feature SHAP values — '
            f'{pred_class} prediction',
            fontsize=10,fontweight='bold')

        patches = [
            mpatches.Patch(
                color='#2563A8',label='Linguistic'),
            mpatches.Patch(
                color='#27AE60',label='Acoustic'),
            mpatches.Patch(
                color='#C0392B',label='CDS'),
        ]
        ax.legend(handles=patches,fontsize=8,
                  loc='lower right')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info(
            "SHAP explanation unavailable. "
            "Install shap: pip install shap")

    # ── Layer 2 — Natural language explanation ─────────
    st.divider()
    st.subheader("💬 Layer 2 — Patient Explanation")

    reasons = generate_explanation(
        feat_scaled[0], shap_vals,
        pred_class, confidence)

    cls_color = cls_colors[pred_class]
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {cls_color};
            padding: 16px 20px;
            border-radius: 8px;
            background: {cls_color}11;
            margin: 8px 0;">
        <p style="margin:0 0 8px;
                  font-weight:600;
                  color:{cls_color}">
            {cls_icons[pred_class]} Predicted:
            {pred_class} — Confidence:
            {confidence:.1%}
        </p>
        <p style="margin:0 0 6px;font-size:14px">
            <strong>Key contributing factors:</strong>
        </p>
        <ul style="margin:0;font-size:14px">
        {"".join(f"<li>{r}</li>" for r in reasons)}
        </ul>
        <p style="margin:12px 0 0;
                  font-size:12px;
                  color:#666;
                  font-style:italic">
            This explanation is generated by TreeSHAP
            on the Random Forest component. It is
            intended to support, not replace,
            clinical assessment.
        </p>
        </div>
        """,
        unsafe_allow_html=True)

    # ── Transcript preview ─────────────────────────────
    st.divider()
    with st.expander(
            "📄 Extracted transcript "
            "(PAR-only Cookie Theft)"):
        st.write(transcript)
        st.caption(
            f"Word count: {len(transcript.split())}")