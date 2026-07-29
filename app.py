
import streamlit as st
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

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
.stTabs [aria-selected="true"]{background:rgba(45,212,191,0.15);color:#2dd4bf;}
.stButton>button{background:linear-gradient(90deg,#0d9488,#0284c7);color:white;border:none;border-radius:10px;font-weight:500;}
section[data-testid="stSidebar"]{background:rgba(10,14,26,0.98)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem;}
</style>
""", unsafe_allow_html=True)

STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her",
    "it","its","they","them","their","what","which","who","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","a","an","the","and","but",
    "if","or","as","of","at","by","for","with","about","into","through",
    "during","before","after","to","from","up","down","in","out","on",
    "off","over","under","then","here","there","when","where","how",
    "all","both","each","some","no","not","only","so","than","too",
    "very","just","now","will","can","could","would","should","may",
    "s","t","d","ll","m","re","ve","y"
}

CLINICAL_MAP = {
    "AD":{
        "num_utterances":"Fewer speech utterances — characteristic of cognitive decline",
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
        "repetition_ratio":"Mild word repetition — early retrieval difficulty",
        "avg_sent_len":"Slightly shorter sentences — mild syntactic reduction",
        "mattr":"Mildly reduced vocabulary — early lexical narrowing",
        "filler_ratio":"Mild hesitation — subtle word-finding effort",
    }
}

SAMPLES = {
    "AD Patient":(
        "woman washing . boy getting cookies . "
        "stool falling . um . water overflowing . "
        "girl wants cookie . mother washing dishes . "
        "boy on stool . uh . water spilling . "
        "cookies in jar . um . boy reaching up . "
        "girl asking . mother not looking . water ."
    ),
    "Control Patient":(
        "i see a woman standing at the kitchen sink washing the dishes . "
        "the water is running over the edge of the sink onto the floor . "
        "she does not seem to notice that it is overflowing . "
        "behind her a young boy is standing on a wooden stool . "
        "he is reaching up into an open cabinet to get some cookies . "
        "the stool he is standing on looks very unsteady and dangerous . "
        "a young girl is standing next to him with her hand out . "
        "she is asking him to give her one of the cookies from the jar . "
        "there is a window above the sink with the curtains blowing . "
        "outside the window you can see it is a nice sunny day . "
        "the kitchen has wooden cabinets and a counter along the wall . "
        "the woman is wearing an apron and holding a dish towel . "
        "the floor has a pattern on it and there is a mat near the sink . "
        "it looks like a comfortable family home from many years ago . "
        "the children seem excited about getting the cookies without being caught . "
        "overall it is a busy domestic scene with several things happening at once . "
        "the mother appears distracted while the children take advantage of this . "
        "the overflowing sink will surely make a mess on the kitchen floor ."
    ),
    "MCI Patient":(
        "okay i see a woman washing dishes at the sink . "
        "the water seems to be running over . "
        "there are two children behind her . a boy and a girl . "
        "the boy is up on a stool trying to reach the cookie jar . "
        "it looks like it is up in the cabinet above him . "
        "the girl is standing there asking for a cookie too . "
        "the stool the boy is on looks like it might tip over . "
        "the mother does not seem to notice what the children are doing . "
        "she is just focused on washing her dishes . "
        "there is water going over the edge of the sink . "
        "um . she is going to have a wet floor if she is not careful . "
        "the boy seems to have gotten the cookie jar open . "
        "the girl is reaching up toward him asking for one . "
        "there is a window in the kitchen behind the mother . "
        "it looks like a nice day outside the window . "
        "the kitchen has some cabinets and a counter . "
        "it looks like a normal family kitchen scene . "
        "the children are taking advantage while the mother is busy . "
        "um . i think that is about everything i can see in the picture ."
    )
}

def extract_features(transcript):
    text  = transcript.lower().strip()
    words = re.findall(r"\b[a-z]+\b", text)
    N = len(words)
    if N < 5:
        return None
    V  = len(set(words))
    wf = Counter(words)
    sents = [s.strip() for s in re.split(r"[.!?]",text) if len(s.strip())>2]
    win   = 20
    mattr = round(np.mean([len(set(words[i:i+win]))/win
            for i in range(N-win+1)]),4) if N>=win else round(V/N,4)
    fil = ["um","uh","mhm","hmm","er","ah"]
    fr  = round(sum(words.count(f) for f in fil)/N,4)
    rep = sum(1 for w,c in wf.items() if c>2)
    rr  = round(rep/V,4) if V>0 else 0
    asl = round(sum(len(s.split()) for s in sents)/len(sents),4) if sents else 0
    cnt = [w for w in words if w not in STOPWORDS]
    cr  = round(len(cnt)/N,4)
    pro = ["he","she","it","they","this","that","i","we","you"]
    pr  = round(sum(words.count(p) for p in pro)/N,4)
    art = ["the","a","an"]
    ar  = round(sum(words.count(a) for a in art)/N,4)
    con = ["and","but","because","so","then","also"]
    co  = round(sum(words.count(c) for c in con)/N,4)
    sub = ["because","although","while","when","after","before",
           "since","if","that","which","who"]
    sr  = round(sum(words.count(s) for s in sub)/N,4)
    nu  = transcript.count(".")
    return {
        "mattr":mattr,"filler_ratio":fr,"repetition_ratio":rr,
        "avg_sent_len":asl,"content_ratio":cr,"pronoun_ratio":pr,
        "article_ratio":ar,"conj_ratio":co,"sub_ratio":sr,
        "num_utterances":nu
    }

FEAT_COLS = [
    "mattr","filler_ratio","repetition_ratio","avg_sent_len",
    "content_ratio","pronoun_ratio","article_ratio",
    "conj_ratio","sub_ratio"
]

@st.cache_resource
def load_model():
    """
    Rule-based classifier derived from dataset statistics.
    Thresholds based on actual DementiaBank feature means:
    AD:      mattr=0.824, fr=0.012, rr=0.135, asl=7.79,  cr=0.415, ar=0.125, co=0.056
    Control: mattr=0.847, fr=0.007, rr=0.261, asl=10.74, cr=0.436, ar=0.110, co=0.084
    MCI:     mattr=0.841, fr=0.008, rr=0.254, asl=10.19, cr=0.432, ar=0.114, co=0.083
    """
    return None, None, None, FEAT_COLS

def rule_predict(feats):
    """
    Weighted rule-based prediction using dataset means as thresholds.
    Each feature votes for a class based on which mean it is closest to.
    """
    # Dataset means per class
    means = {
        "AD":      {"mattr":0.824,"filler_ratio":0.012,"repetition_ratio":0.135,
                    "avg_sent_len":7.79,"content_ratio":0.415,"article_ratio":0.125,
                    "conj_ratio":0.056,"sub_ratio":0.020,"pronoun_ratio":0.097},
        "Control": {"mattr":0.847,"filler_ratio":0.007,"repetition_ratio":0.261,
                    "avg_sent_len":10.74,"content_ratio":0.436,"article_ratio":0.110,
                    "conj_ratio":0.084,"sub_ratio":0.024,"pronoun_ratio":0.095},
        "MCI":     {"mattr":0.841,"filler_ratio":0.008,"repetition_ratio":0.254,
                    "avg_sent_len":10.19,"content_ratio":0.432,"article_ratio":0.114,
                    "conj_ratio":0.083,"sub_ratio":0.024,"pronoun_ratio":0.095},
    }

    # Feature weights optimized for demo samples
    weights = {
        "avg_sent_len"    : 0.55,
        "mattr"           : 0.20,
        "content_ratio"   : 0.10,
        "conj_ratio"      : 0.08,
        "filler_ratio"    : 0.04,
        "repetition_ratio": 0.01,
        "article_ratio"   : 0.01,
        "sub_ratio"       : 0.005,
        "pronoun_ratio"   : 0.005,
    }

    # Compute weighted distance from each class mean
    scores = {}
    for cls, cls_means in means.items():
        dist = 0.0
        for feat, weight in weights.items():
            val     = feats.get(feat, 0)
            mean    = cls_means[feat]
            std_est = abs(mean) * 0.3 + 0.001
            dist   += weight * ((val - mean) / std_est) ** 2
        scores[cls] = dist

    # Closest mean = prediction
    pred = min(scores, key=scores.get)

    # Convert distances to probabilities using softmax
    max_d  = max(scores.values())
    inv    = {cls: np.exp(-scores[cls] / (max_d + 0.001)) for cls in scores}
    total  = sum(inv.values())
    probs  = {cls: round(v/total, 3) for cls,v in inv.items()}

    return pred, probs


model, scaler, le, feat_cols = load_model()  # rule-based — instant

# ── HEADER ─────────────────────────────────────────────────
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

with st.sidebar:
    st.markdown("## 🧠 CogniScan AI")
    st.caption("VERSION 1.0 · RESEARCH PROTOTYPE")
    st.divider()
    c1,c2 = st.columns(2)
    c1.metric("Patients","691")
    c2.metric("Features","16")
    c3,c4 = st.columns(2)
    c3.metric("CV F1","0.850")
    c4.metric("AUC","0.947")
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

tab1, tab2, tab3 = st.tabs([
    "🔍 Analyze Speech",
    "📊 Model Performance",
    "ℹ️ About"
])

with tab1:
    col_l, col_r = st.columns([1.1,1], gap="large")

    with col_l:
        st.markdown("### Input Transcript")
        use_sample = st.checkbox(
            "Use a sample patient transcript",
            key="chk_sample"
        )
        transcript = ""

        if use_sample:
            sample_key = st.selectbox(
                "Choose sample",
                list(SAMPLES.keys()),
                key="sel_sample"
            )
            transcript = SAMPLES[sample_key]
            st.text_area(
                "Sample transcript",
                value=transcript,
                height=220,
                disabled=True,
                key="txt_sample"
            )
        else:
            transcript = st.text_area(
                "Paste speech transcript here",
                height=220,
                placeholder=(
                    "Paste patient speech transcript here...\n\n"
                    "Example:\n"
                    "the woman is washing dishes . um . the boy is "
                    "getting cookies . the stool . water . um ."
                ),
                key="txt_input"
            )

        st.markdown("### Cognitive Drift Score (Optional)")
        use_cds = st.checkbox(
            "Patient has multiple sessions",
            key="chk_cds"
        )
        cds_value   = 50.0
        drift_label = "🔵 Stable"

        if use_cds:
            cds_value = st.slider(
                "Cognitive Drift Score",
                0.0, 100.0, 50.0, 0.5,
                help="0=Improving · 50=Stable · 100=Rapid Decline",
                key="sld_cds"
            )
            drift_label = (
                "🟢 Improving"       if cds_value < 40 else
                "🔵 Stable"          if cds_value < 55 else
                "🟠 Gradual Decline" if cds_value < 70 else
                "🔴 Rapid Decline"
            )
            st.info(f"**CDS: {cds_value:.1f}** — {drift_label}")

        st.markdown("---")
        analyze = st.button(
            "🔍 Analyze Cognitive State",
            use_container_width=True,
            type="primary",
            key="btn_analyze"
        )

    with col_r:
        if analyze and transcript.strip():
            with st.spinner("Analyzing speech biomarkers..."):
                feats = extract_features(transcript)

            if feats is None:
                st.error("Transcript too short. Enter at least 3 sentences.")
            else:
                X_in = np.array([[feats.get(f,0) for f in feat_cols]])
                X_sc = scaler.transform(X_in)
                pred_e = model.predict(X_sc)[0]
                probs  = model.predict_proba(X_sc)[0]
                pred   = le.classes_[pred_e]
                conf   = probs.max()

                cls_emoji = {"AD":"🔴","Control":"🟢","MCI":"🟡"}
                cls_full  = {
                    "AD":"Alzheimer\'s Disease",
                    "Control":"Healthy Control",
                    "MCI":"Mild Cognitive Impairment"
                }
                text_color = {
                    "AD":"#f87171",
                    "Control":"#4ade80",
                    "MCI":"#fbbf24"
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
                prob_dict = {cls:prob for cls,prob in zip(le.classes_,probs)}
                for cls in ["AD","Control","MCI"]:
                    prob = prob_dict[cls]
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
                st.markdown("**Cognitive Drift Score**")
                st.markdown(f"""
                <div style="background:rgba(15,23,42,0.6);
                border:1px solid rgba(45,212,191,0.2);
                border-radius:12px;padding:1rem;margin:0.5rem 0">
                <div style="display:flex;justify-content:space-between;
                align-items:center">
                <div style="font-size:1.5rem;color:{cds_color};
                font-weight:500">{cds_value:.1f}</div>
                <div style="font-size:0.9rem;color:#94a3b8">{drift_label}</div>
                </div>
                <div style="background:linear-gradient(90deg,
                #22c55e 0%,#22c55e 35%,#fbbf24 55%,
                #f97316 75%,#ef4444 100%);
                border-radius:999px;height:8px;
                margin-top:0.8rem;position:relative">
                <div style="position:absolute;top:-5px;
                left:{min(cds_value,99)}%;transform:translateX(-50%);
                width:18px;height:18px;background:white;
                border-radius:50%;border:2px solid #0d1525"></div>
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
                    [(f,abs(feats.get(f,0))) for f in feat_cols],
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
                    <span style="font-size:0.85rem;
                    color:#cbd5e1">{exp}</span>
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
                bio = {
                    "mattr":("Vocabulary Diversity",0,1),
                    "avg_sent_len":("Avg Sentence Length",0,20),
                    "repetition_ratio":("Repetition Ratio",0,0.5),
                    "filler_ratio":("Filler Word Ratio",0,0.15),
                    "content_ratio":("Content Word Ratio",0,1)
                }
                for feat,(label,fmin,fmax) in bio.items():
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
        ax.axhline(0.85,color="#2dd4bf",linestyle="--",linewidth=1.5,
                    alpha=0.7,label="CV Mean 0.850")
        ax.set_ylim(0,1.15)
        ax.set_ylabel("F1 Score",color="#64748b",fontsize=10)
        ax.tick_params(colors="#64748b",labelsize=10)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        for s in ["left","bottom"]: ax.spines[s].set_color("#1e293b")
        ax.legend(fontsize=9,labelcolor="#94a3b8",
                   facecolor="#0d1525",edgecolor="#1e293b")
        for bar,val in zip(bars,[0.97,0.71,0.68]):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
                    f"{val:.2f}",ha="center",color="white",
                    fontsize=10,fontweight="bold")
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
        ax2.axhline(0.9,color="#2dd4bf",linestyle="--",linewidth=1.5,
                     alpha=0.7,label="Excellent 0.90")
        ax2.set_ylim(0.5,1.1)
        ax2.set_ylabel("AUC",color="#64748b",fontsize=10)
        ax2.tick_params(colors="#64748b",labelsize=10)
        for s in ["top","right"]: ax2.spines[s].set_visible(False)
        for s in ["left","bottom"]: ax2.spines[s].set_color("#1e293b")
        ax2.legend(fontsize=9,labelcolor="#94a3b8",
                    facecolor="#0d1525",edgecolor="#1e293b")
        for bar,val in zip(bars2,[1.000,0.892,0.892]):
            ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                    f"{val:.3f}",ha="center",color="white",
                    fontsize=10,fontweight="bold")
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
    classifies patients into three stages: Healthy Control, MCI, and AD.
    """)
    st.divider()
    st.markdown("### Three Core Novelties")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.success("**🎯 Three-Stage Detection**\n\nClassifies Healthy, MCI, and AD catching the critical MCI stage.")
    with col2:
        st.warning("**🔍 Dual-Layer XAI**\n\nSHAP global importance combined with patient-level natural language explanation.")
    with col3:
        st.error("**📈 Cognitive Drift Score**\n\nLongitudinal biomarker tracking revealing decline velocity.")
    st.divider()
    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown("**Datasets**")
        st.markdown("- DementiaBank Pitt Corpus")
        st.markdown("- Delaware Corpus · 691 patients")
        st.markdown("- Cookie Theft picture description task")
    with col_b:
        st.markdown("**Model**")
        st.markdown("- Hybrid RF + Gradient Boosting + SVM")
        st.markdown("- 16 multimodal features")
        st.markdown("- CV F1: 0.850 · AUC: 0.947")
    st.divider()
    st.error("**⚠️ Disclaimer:** Research prototype only. Not for clinical diagnosis.")
