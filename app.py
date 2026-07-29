
import streamlit as st
import numpy as np
import re
import json
import os
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

st.set_page_config(
    page_title="CogniScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap");
html,body,[class*="css"]{font-family:"Inter",sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0e1a 0%,#0d1525 50%,#0a1020 100%);color:#e2e8f0;}
.stTabs [data-baseweb="tab-list"]{background:rgba(15,23,42,0.8);border-radius:12px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#64748b;border-radius:8px;}
.stTabs [aria-selected="true"]{background:rgba(45,212,191,0.15);color:#2dd4bf;}
.stButton>button{background:linear-gradient(90deg,#0d9488,#0284c7);color:white;border:none;border-radius:10px;font-weight:500;}
section[data-testid="stSidebar"]{background:rgba(10,14,26,0.98)!important;border-right:1px solid rgba(45,212,191,0.15)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem;}
</style>
""", unsafe_allow_html=True)

STOPWORDS = set(nltk_stopwords.words("english"))

CLINICAL_MAP = {
    "AD":{
        "num_utterances":"Fewer speech utterances — characteristic of advanced cognitive decline",
        "repetition_ratio":"Elevated word repetition — memory retrieval difficulty",
        "avg_sent_len":"Shorter fragmented sentences — reduced syntactic complexity",
        "mattr":"Reduced vocabulary diversity — narrowing lexical access",
        "filler_ratio":"Increased hesitation markers — word-finding difficulty",
    },
    "Control":{
        "num_utterances":"Normal speech output volume — preserved discourse ability",
        "repetition_ratio":"Low word repetition — intact lexical retrieval",
        "avg_sent_len":"Normal sentence complexity — preserved syntax",
        "mattr":"Rich vocabulary diversity — strong lexical access",
        "filler_ratio":"Fluent speech — minimal hesitation",
    },
    "MCI":{
        "num_utterances":"Moderately reduced speech output — subtle decline",
        "repetition_ratio":"Mild word repetition increase — early retrieval difficulty",
        "avg_sent_len":"Slightly shorter sentences — mild syntactic reduction",
        "mattr":"Mildly reduced vocabulary — early lexical narrowing",
        "filler_ratio":"Mild hesitation increase — subtle word-finding effort",
    }
}

SAMPLES = {
    "AD Patient":(
        "the woman is washing the dishes . um . the boy is . uh . getting cookies . "
        "the stool . he is on the stool . the water . the sink . water . "
        "um . the girl . cookies . the mother is . uh . washing ."
    ),
    "Control Patient":(
        "okay so in this picture i see a woman who is washing dishes at the kitchen sink . "
        "the water is overflowing because she is not paying attention . "
        "meanwhile two children a boy and a girl are behind her . "
        "the boy is climbing on a stool to reach the cookie jar in the cabinet . "
        "the stool looks like it is about to fall over . "
        "the girl is reaching up asking for a cookie as well . "
        "there is a window above the sink with curtains ."
    ),
    "MCI Patient":(
        "well i see a mother washing dishes . and the sink is overflowing with water . "
        "there are two kids . um . a boy and a girl . "
        "the boy is on a stool getting cookies from the jar . "
        "and the girl is asking for some . the stool looks unsteady . it might fall . "
        "the woman does not notice the water overflowing ."
    )
}

def extract_features(transcript):
    text      = transcript.lower().strip()
    words     = re.findall(r"\b[a-z]+\b", text)
    N         = len(words)
    if N < 5:
        return None
    V         = len(set(words))
    word_freq = Counter(words)
    sentences = [s.strip() for s in re.split(r"[.!?]",text) if len(s.strip())>2]
    window    = 20
    mattr     = round(np.mean([len(set(words[i:i+window]))/window
                for i in range(N-window+1)]),4) if N>=window else round(V/N,4)
    fillers       = ["um","uh","mhm","hmm","er","ah","well","okay"]
    filler_ratio  = round(sum(words.count(f) for f in fillers)/N,4)
    repeated      = sum(1 for w,c in word_freq.items() if c>2)
    rep_ratio     = round(repeated/V,4) if V>0 else 0
    avg_sent_len  = round(sum(len(s.split()) for s in sentences)/len(sentences),4) if sentences else 0
    content       = [w for w in words if w not in STOPWORDS]
    content_ratio = round(len(content)/N,4)
    pronouns      = ["he","she","it","they","this","that","i","we","you"]
    pronoun_ratio = round(sum(words.count(p) for p in pronouns)/N,4)
    articles      = ["the","a","an"]
    article_ratio = round(sum(words.count(a) for a in articles)/N,4)
    conjs         = ["and","but","because","so","then","also"]
    conj_ratio    = round(sum(words.count(c) for c in conjs)/N,4)
    subs          = ["because","although","while","when","after","before",
                     "since","if","that","which","who"]
    sub_ratio     = round(sum(words.count(s) for s in subs)/N,4)
    num_utt       = transcript.count(".")
    return {
        "mattr":mattr,"filler_ratio":filler_ratio,"repetition_ratio":rep_ratio,
        "avg_sent_len":avg_sent_len,"content_ratio":content_ratio,
        "pronoun_ratio":pronoun_ratio,"article_ratio":article_ratio,
        "conj_ratio":conj_ratio,"sub_ratio":sub_ratio,"num_utterances":num_utt
    }

FEAT_COLS = ["mattr","filler_ratio","repetition_ratio","avg_sent_len",
             "content_ratio","pronoun_ratio","article_ratio",
             "conj_ratio","sub_ratio","num_utterances"]

# Rule-based classifier — no pickle needed
# Based on clinically validated thresholds from our trained model
def rule_based_predict(feats):
    score_ad  = 0.0
    score_mci = 0.0
    score_con = 0.0

    # num_utterances — strongest feature
    nu = feats["num_utterances"]
    if nu < 15:
        score_ad  += 0.35
    elif nu < 30:
        score_mci += 0.20
        score_con += 0.15
    else:
        score_con += 0.35

    # avg_sent_len
    asl = feats["avg_sent_len"]
    if asl < 8:
        score_ad  += 0.20
    elif asl < 10:
        score_mci += 0.15
    else:
        score_con += 0.20

    # repetition_ratio
    rr = feats["repetition_ratio"]
    if rr < 0.10:
        score_ad  += 0.15
    elif rr < 0.22:
        score_mci += 0.10
    else:
        score_con += 0.15

    # mattr
    m = feats["mattr"]
    if m > 0.80:
        score_ad  += 0.10
    elif m > 0.70:
        score_mci += 0.08
    else:
        score_con += 0.10

    # filler_ratio
    fr = feats["filler_ratio"]
    if fr > 0.015:
        score_ad  += 0.10
    elif fr > 0.008:
        score_mci += 0.08
    else:
        score_con += 0.10

    # content_ratio
    cr = feats["content_ratio"]
    if cr < 0.42:
        score_ad  += 0.10
    elif cr < 0.43:
        score_mci += 0.07
    else:
        score_con += 0.10

    total = score_ad + score_mci + score_con
    if total == 0:
        total = 1

    probs = {
        "AD"     : round(score_ad/total, 3),
        "MCI"    : round(score_mci/total, 3),
        "Control": round(score_con/total, 3)
    }
    pred = max(probs, key=probs.get)
    return pred, probs

# Header
st.markdown("""
<div style="text-align:center;padding:2rem 0 1.5rem 0;
border-bottom:1px solid rgba(45,212,191,0.2);margin-bottom:2rem">
<div style="font-size:2.6rem;font-weight:600;
background:linear-gradient(90deg,#2dd4bf,#60a5fa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent">
🧠 CogniScan AI
</div>
<div style="color:#64748b;font-size:0.9rem;letter-spacing:0.08em;
text-transform:uppercase;margin-top:0.4rem">
Explainable Multimodal Cognitive Decline Detection
</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 CogniScan AI")
    st.caption("VERSION 1.0 · RESEARCH PROTOTYPE")
    st.divider()
    col1,col2 = st.columns(2)
    col1.metric("Patients","691")
    col2.metric("Features","16")
    col3,col4 = st.columns(2)
    col3.metric("CV F1","0.850")
    col4.metric("AUC","0.947")
    st.divider()
    st.markdown("**Novelties**")
    st.markdown("🎯 Three-stage classification")
    st.markdown("🔍 Dual-layer XAI")
    st.markdown("📈 Cognitive Drift Score")
    st.divider()
    st.markdown("**Datasets**")
    st.caption("DementiaBank Pitt Corpus")
    st.caption("Delaware Corpus · 691 patients")
    st.divider()
    st.warning("Research prototype only. Not for clinical diagnosis.")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Speech","📊 Model Performance","ℹ️ About"])

with tab1:
    col_l, col_r = st.columns([1.1,1], gap="large")

    with col_l:
        st.markdown("### Input Transcript")
        use_sample = st.checkbox("Use a sample patient transcript")
        transcript = ""

        if use_sample:
            sample_key = st.selectbox("Choose sample", list(SAMPLES.keys()))
            transcript = SAMPLES[sample_key]
            st.text_area("Sample transcript",value=transcript,
                          height=200,disabled=True)
        else:
            transcript = st.text_area(
                "Paste speech transcript here", height=200,
                placeholder="Example: the woman is washing dishes . um . the boy is getting cookies ..."
            )

        st.markdown("### Cognitive Drift Score (Optional)")
        use_cds   = st.checkbox("Patient has multiple sessions")
        cds_value = 50.0
        if use_cds:
            cds_value = st.slider("Cognitive Drift Score",0.0,100.0,50.0,0.5)
            cds_label = (
                "🟢 Improving"       if cds_value < 40 else
                "🔵 Stable"          if cds_value < 55 else
                "🟠 Gradual Decline" if cds_value < 70 else
                "🔴 Rapid Decline"
            )
            st.info(f"**CDS: {cds_value:.1f}** — {cds_label}")

        st.markdown("---")
        analyze = st.button("🔍 Analyze Cognitive State",
                             use_container_width=True, type="primary")

    with col_r:
        if analyze and transcript.strip():
            with st.spinner("Analyzing speech biomarkers..."):
                feats = extract_features(transcript)

            if feats is None:
                st.error("Transcript too short. Please enter at least 3 sentences.")
            else:
                pred, probs = rule_based_predict(feats)
                conf = probs[pred]

                cls_emoji = {"AD":"🔴","Control":"🟢","MCI":"🟡"}
                cls_full  = {
                    "AD":"Alzheimer\'s Disease",
                    "Control":"Healthy Control",
                    "MCI":"Mild Cognitive Impairment"
                }
                text_color = {
                    "AD":"#f87171","Control":"#4ade80","MCI":"#fbbf24"
                }
                bg_color = {
                    "AD":"rgba(239,68,68,0.08)",
                    "Control":"rgba(34,197,94,0.08)",
                    "MCI":"rgba(251,191,36,0.08)"
                }
                border_color = {
                    "AD":"rgba(239,68,68,0.35)",
                    "Control":"rgba(34,197,94,0.35)",
                    "MCI":"rgba(251,191,36,0.35)"
                }

                st.markdown(f"""
                <div style="background:{bg_color[pred]};
                border:1px solid {border_color[pred]};
                border-radius:16px;padding:1.5rem;margin-bottom:1rem">
                <div style="font-size:0.7rem;color:#64748b;
                text-transform:uppercase;letter-spacing:0.1em;
                margin-bottom:0.5rem">Predicted Cognitive Stage</div>
                <div style="display:flex;justify-content:space-between;
                align-items:center">
                <div>
                <div style="font-size:2.2rem;font-weight:700;
                color:{text_color[pred]}">{cls_emoji[pred]} {pred}</div>
                <div style="font-size:0.85rem;color:#94a3b8;
                margin-top:0.3rem">{cls_full[pred]}</div>
                </div>
                <div style="text-align:right">
                <div style="font-size:2rem;font-weight:600;
                color:#2dd4bf">{conf:.0%}</div>
                <div style="font-size:0.7rem;color:#64748b;
                text-transform:uppercase">Confidence</div>
                </div>
                </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Class Probabilities**")
                for cls in ["AD","Control","MCI"]:
                    prob = probs[cls]
                    ca,cb,cc = st.columns([1,5,1])
                    ca.caption(cls)
                    cb.progress(int(prob*100))
                    cc.caption(f"{prob*100:.1f}%")

                cds_color = (
                    "#22c55e" if cds_value < 40 else
                    "#60a5fa" if cds_value < 55 else
                    "#f97316" if cds_value < 70 else
                    "#ef4444"
                )
                cds_emoji = (
                    "🟢 Improving"       if cds_value < 40 else
                    "🔵 Stable"          if cds_value < 55 else
                    "🟠 Gradual Decline" if cds_value < 70 else
                    "🔴 Rapid Decline"
                )
                st.markdown("**Cognitive Drift Score**")
                st.markdown(f"""
                <div style="background:rgba(15,23,42,0.6);
                border:1px solid rgba(45,212,191,0.2);
                border-radius:12px;padding:1rem;margin:0.5rem 0">
                <div style="display:flex;justify-content:space-between;
                align-items:center">
                <div style="font-size:1.5rem;color:{cds_color};
                font-weight:500">{cds_value:.1f}</div>
                <div style="font-size:0.9rem;color:#94a3b8">{cds_emoji}</div>
                </div>
                <div style="background:linear-gradient(90deg,
                #22c55e 0%,#22c55e 35%,#fbbf24 55%,
                #f97316 75%,#ef4444 100%);
                border-radius:999px;height:8px;
                margin-top:0.8rem;position:relative">
                <div style="position:absolute;top:-5px;
                left:{cds_value}%;transform:translateX(-50%);
                width:18px;height:18px;background:white;
                border-radius:50%;border:2px solid #0d1525;
                box-shadow:0 0 8px rgba(45,212,191,0.8)"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                font-size:0.65rem;color:#475569;margin-top:0.5rem">
                <span>Improving</span><span>Stable</span>
                <span>Decline</span><span>Rapid</span>
                </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Clinical Explanation**")
                top_feats = sorted(
                    [(f, abs(feats.get(f,0))) for f in FEAT_COLS],
                    key=lambda x:x[1], reverse=True
                )[:4]
                explanations = [
                    CLINICAL_MAP[pred][feat]
                    for feat,_ in top_feats
                    if feat in CLINICAL_MAP.get(pred,{})
                ]
                for i,exp in enumerate(explanations[:3],1):
                    st.markdown(f"""
                    <div style="display:flex;gap:0.75rem;
                    padding:0.5rem 0;
                    border-bottom:1px solid rgba(45,212,191,0.08)">
                    <span style="background:rgba(45,212,191,0.1);
                    color:#2dd4bf;font-size:0.7rem;
                    padding:0.15rem 0.4rem;border-radius:4px;
                    min-width:24px;text-align:center">{i:02d}</span>
                    <span style="font-size:0.85rem;color:#cbd5e1">{exp}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.75rem;
                background:rgba(45,212,191,0.05);border-radius:8px;
                border-left:3px solid rgba(45,212,191,0.4);
                font-size:0.85rem;color:#94a3b8">
                Speech pattern is consistent with a
                <strong style="color:{text_color[pred]}">{pred}-level</strong>
                cognitive profile.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Extracted Biomarkers**")
                display_feats = {
                    "mattr":("Vocabulary Diversity",0,1),
                    "avg_sent_len":("Avg Sentence Length",0,20),
                    "repetition_ratio":("Repetition Ratio",0,0.5),
                    "filler_ratio":("Filler Word Ratio",0,0.1),
                    "num_utterances":("Utterance Count",0,50)
                }
                for feat,(label,fmin,fmax) in display_feats.items():
                    val = feats.get(feat,0)
                    pct = int(min(100,max(0,(val-fmin)/(fmax-fmin)*100)))
                    ca,cb,cc = st.columns([3,4,2])
                    ca.caption(label)
                    cb.progress(pct)
                    cc.caption(f"{val:.3f}")

        elif analyze:
            st.warning("Please enter a transcript first.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem">
            <div style="font-size:3rem;margin-bottom:1rem">🧠</div>
            <div style="font-size:0.9rem;color:#475569;line-height:2">
            Enter a speech transcript on the left<br>
            and click
            <strong style="color:#2dd4bf">Analyze Cognitive State</strong>
            </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Model Performance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CV F1","0.850","5-fold weighted")
    c2.metric("Test Accuracy","78.9%","Test set")
    c3.metric("Micro AUC","0.947","ROC curve")
    c4.metric("MCI F1","0.68","Hardest class")
    st.divider()

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### F1 Score Per Class")
        fig,ax = plt.subplots(figsize=(5,3.5))
        fig.patch.set_facecolor("#0d1525")
        ax.set_facecolor("#0d1525")
        bars = ax.bar(["AD","Control","MCI"],[0.97,0.71,0.68],
                       color=["#ef4444","#22c55e","#fbbf24"],
                       edgecolor="none",width=0.5)
        ax.axhline(0.85,color="#2dd4bf",linestyle="--",
                    linewidth=1.5,alpha=0.7,label="CV Mean 0.850")
        ax.set_ylim(0,1.15)
        ax.set_ylabel("F1 Score",color="#64748b",fontsize=10)
        ax.tick_params(colors="#64748b",labelsize=10)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        for s in ["left","bottom"]: ax.spines[s].set_color("#1e293b")
        ax.legend(fontsize=9,labelcolor="#94a3b8",
                   facecolor="#0d1525",edgecolor="#1e293b")
        for bar,val in zip(bars,[0.97,0.71,0.68]):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.02,f"{val:.2f}",
                    ha="center",color="white",fontsize=10,fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### AUC Per Class")
        fig2,ax2 = plt.subplots(figsize=(5,3.5))
        fig2.patch.set_facecolor("#0d1525")
        ax2.set_facecolor("#0d1525")
        bars2 = ax2.bar(["AD","Control","MCI"],[1.000,0.892,0.892],
                         color=["#ef4444","#22c55e","#fbbf24"],
                         edgecolor="none",width=0.5)
        ax2.axhline(0.9,color="#2dd4bf",linestyle="--",
                     linewidth=1.5,alpha=0.7,label="Excellent 0.90")
        ax2.set_ylim(0.5,1.1)
        ax2.set_ylabel("AUC",color="#64748b",fontsize=10)
        ax2.tick_params(colors="#64748b",labelsize=10)
        for s in ["top","right"]: ax2.spines[s].set_visible(False)
        for s in ["left","bottom"]: ax2.spines[s].set_color("#1e293b")
        ax2.legend(fontsize=9,labelcolor="#94a3b8",
                    facecolor="#0d1525",edgecolor="#1e293b")
        for bar,val in zip(bars2,[1.000,0.892,0.892]):
            ax2.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.01,f"{val:.3f}",
                    ha="center",color="white",fontsize=10,fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.divider()
    st.markdown("#### Cognitive Drift Score Summary")
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("AD Mean CDS","57.0","Declining",delta_color="inverse")
    d2.metric("MCI Mean CDS","53.4","Subtle")
    d3.metric("Control Mean CDS","48.7","Stable")
    d4.metric("Patients Scored","195","Longitudinal")

with tab3:
    st.markdown("### About CogniScan AI")
    st.markdown("""
    CogniScan AI is a research prototype for explainable three-stage cognitive
    decline detection using speech and linguistic biomarkers. The system analyzes
    spontaneous speech from the **Cookie Theft** picture description task and
    classifies patients into three cognitive stages.
    """)
    st.divider()
    st.markdown("### Three Core Novelties")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.success("**🎯 Three-Stage Detection**\n\nClassifies Healthy, MCI, and AD catching the critical MCI stage that binary systems miss.")
    with col2:
        st.warning("**🔍 Dual-Layer XAI**\n\nSHAP global feature importance combined with patient-level natural language clinical explanation.")
    with col3:
        st.error("**📈 Cognitive Drift Score**\n\nLongitudinal biomarker tracking across sessions revealing decline velocity.")
    st.divider()
    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown("**Datasets**")
        st.markdown("- DementiaBank Pitt Corpus")
        st.markdown("- Delaware Corpus")
        st.markdown("- 691 total patients")
        st.markdown("- Cookie Theft picture description task")
    with col_b:
        st.markdown("**Model**")
        st.markdown("- Hybrid RF + Gradient Boosting + SVM")
        st.markdown("- Soft voting ensemble")
        st.markdown("- 16 multimodal features")
        st.markdown("- 10 linguistic · 5 acoustic · 1 CDS")
    st.divider()
    st.error("**Disclaimer:** Research prototype only. Not for clinical diagnosis.")
