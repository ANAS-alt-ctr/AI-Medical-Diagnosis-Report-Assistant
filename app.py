import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import base64
from PIL import Image
import io

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI — AI Diagnosis Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Dark Cyberpunk-Medical Theme) ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=Noto+Nastaliq+Urdu&display=swap');

:root {
    --bg-primary:   #080C10;
    --bg-secondary: #0D1117;
    --bg-card:      #111820;
    --bg-glass:     rgba(0, 212, 255, 0.04);
    --accent:       #00D4FF;
    --accent-dim:   rgba(0, 212, 255, 0.15);
    --accent2:      #00FF88;
    --danger:       #FF4444;
    --warning:      #FFB830;
    --text-primary: #E8EDF3;
    --text-muted:   #6B7A8D;
    --border:       rgba(0, 212, 255, 0.12);
    --glow:         0 0 20px rgba(0, 212, 255, 0.25);
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-dim); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
}

/* ── Header Banner ── */
.med-header {
    background: linear-gradient(135deg, #0D1117 0%, #111820 50%, #0a1520 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.med-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 70% 50%, rgba(0,212,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.med-header h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
    background: linear-gradient(90deg, #00D4FF, #00FF88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.med-header p {
    color: var(--text-muted) !important;
    margin: 0.25rem 0 0 0;
    font-size: 0.9rem;
}

/* ── Cards ── */
.med-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    transition: border-color 0.3s;
}
.med-card:hover { border-color: rgba(0,212,255,0.3); }
.med-card h3 {
    color: var(--accent) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem !important;
}

/* ── Risk Badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
}
.risk-low      { background: rgba(0,255,136,0.15); color: #00FF88; border: 1px solid rgba(0,255,136,0.4); }
.risk-medium   { background: rgba(255,184,48,0.15); color: #FFB830; border: 1px solid rgba(255,184,48,0.4); }
.risk-high     { background: rgba(255,100,60,0.15); color: #FF643C; border: 1px solid rgba(255,100,60,0.4); }
.risk-critical { background: rgba(255,68,68,0.15); color: #FF4444; border: 1px solid rgba(255,68,68,0.4); }

/* ── Metric boxes ── */
.metric-box {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-box .metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
    color: var(--accent);
}
.metric-box .metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,184,48,0.06);
    border: 1px solid rgba(255,184,48,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #FFB830;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Urdu text ── */
.urdu-text {
    font-family: 'Noto Nastaliq Urdu', serif;
    direction: rtl;
    text-align: right;
    line-height: 2.2;
    font-size: 1.05rem;
    color: var(--text-primary);
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

/* ── AI Response ── */
.ai-response {
    background: linear-gradient(135deg, #0a1520, #111820);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF22, #00D4FF11) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #080C10 !important;
    box-shadow: var(--glow) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-glass) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
}

/* ── Chat messages ── */
.chat-user {
    background: var(--accent-dim);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.9rem;
}
.chat-ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.9rem;
}

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "vision_result":   None,
        "report_result":   None,
        "risk_result":     None,
        "ai_explanation":  None,
        "patient_info":    {},
        "chat_history":    [],
        "uploaded_image":  None,
        "uploaded_pdf":    None,
        "analysis_done":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem'>🧬</div>
        <div style='font-size:1.1rem; font-weight:800; color:#00D4FF; letter-spacing:-0.02em'>MedAI</div>
        <div style='font-size:0.75rem; color:#6B7A8D'>AI Diagnosis Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🔑 API Configuration")
    nvidia_key = st.text_input(
        "NVIDIA API Key",
        type="password",
        placeholder="nvapi-...",
        help="Get free key at build.nvidia.com"
    )
    if nvidia_key:
        os.environ["NVIDIA_API_KEY"] = nvidia_key
        st.success("✓ API key saved", icon="✅")

    st.divider()

    st.markdown("### 👤 Patient Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    duration = st.selectbox(
        "Symptom Duration",
        ["< 1 week", "1-2 weeks", "2-4 weeks", "1-3 months", "> 3 months"]
    )

    st.markdown("**Symptoms** (select all that apply)")
    symptom_options = [
        "Fever", "Cough", "Shortness of breath", "Chest pain",
        "Fatigue", "Headache", "Nausea", "Skin rash",
        "Joint pain", "Weight loss", "Night sweats", "Dizziness"
    ]
    selected_symptoms = st.multiselect(
        "Symptoms",
        symptom_options,
        label_visibility="collapsed"
    )

    st.session_state.patient_info = {
        "age": age,
        "gender": gender,
        "duration": duration,
        "symptoms": selected_symptoms,
    }

    st.divider()
    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <span>Educational tool only. Always consult a qualified physician.</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='med-header'>
    <h1>🧬 MedAI — Intelligent Diagnosis Assistant</h1>
    <p>AI-powered medical image analysis · PDF report reading · Risk prediction · Bilingual AI doctor</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📤  Upload & Analyse",
    "🔬  Diagnosis Results",
    "📊  Health Dashboard",
    "💬  AI Doctor Chat",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Analyse
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("<div class='med-card'><h3>📁 Upload Medical File</h3>", unsafe_allow_html=True)
        upload_type = st.radio(
            "Input type",
            ["🩻 X-Ray / Medical Image", "🔬 Skin Disease Photo", "📄 PDF Medical Report"],
            horizontal=True,
        )

        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=["jpg", "jpeg", "png", "pdf"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            ext = Path(uploaded_file.name).suffix.lower()
            if ext == ".pdf":
                st.session_state.uploaded_pdf    = uploaded_file
                st.session_state.uploaded_image  = None
                st.info(f"📄 PDF loaded: **{uploaded_file.name}**")
            else:
                st.session_state.uploaded_image  = uploaded_file
                st.session_state.uploaded_pdf    = None
                img = Image.open(uploaded_file)
                st.image(img, caption=uploaded_file.name, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='med-card'><h3>👤 Patient Summary</h3>", unsafe_allow_html=True)
        pi = st.session_state.patient_info
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{pi.get('age','—')}</div><div class='metric-label'>Age</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{pi.get('gender','—')[:1]}</div><div class='metric-label'>Gender</div></div>", unsafe_allow_html=True)
        with c3:
            dur_map = {"< 1 week":"<1W","1-2 weeks":"1-2W","2-4 weeks":"2-4W","1-3 months":"1-3M","> 3 months":">3M"}
            dur_short = dur_map.get(pi.get("duration",""),"—")
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{dur_short}</div><div class='metric-label'>Duration</div></div>", unsafe_allow_html=True)

        if pi.get("symptoms"):
            st.markdown("<br>**Active Symptoms:**", unsafe_allow_html=True)
            chips = " ".join([f"<span style='background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);padding:2px 10px;border-radius:20px;font-size:0.78rem;margin:2px;display:inline-block;color:#00D4FF'>{s}</span>" for s in pi["symptoms"]])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown("<br><span style='color:#6B7A8D;font-size:0.85rem'>No symptoms selected yet.</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Run Analysis Button ──
    st.divider()
    btn_col = st.columns([1, 2, 1])[1]
    with btn_col:
        run_analysis = st.button("🚀  Run Full AI Analysis", use_container_width=True)

    if run_analysis:
        if not uploaded_file:
            st.warning("⚠️ Please upload an image or PDF first.")
        else:
            with st.status("🔄 Running AI analysis pipeline…", expanded=True) as status:

                # Step 1 — Vision
                if st.session_state.uploaded_image:
                    st.write("🩻 Step 1/4 — Scanning image with YOLOv11…")
                    try:
                        from vision_model import analyze_image
                        with tempfile.NamedTemporaryFile(
                            suffix=Path(uploaded_file.name).suffix, delete=False
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        st.session_state.vision_result = analyze_image(tmp_path)
                        os.unlink(tmp_path)
                        st.write(f"   ✅ Detected: **{st.session_state.vision_result.get('disease','Unknown')}** "
                                 f"({st.session_state.vision_result.get('confidence', 0):.1%} confidence)")
                    except Exception as e:
                        st.warning(f"   ⚠️ Vision model: {e}")
                        st.session_state.vision_result = {
                            "disease": "Unknown (model not loaded)",
                            "confidence": 0.0,
                            "annotated_image": None,
                        }

                # Step 2 — NLP PDF
                if st.session_state.uploaded_pdf:
                    st.write("📄 Step 1/4 — Extracting report with spaCy NLP…")
                    try:
                        from report_reader import extract_report
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        st.session_state.report_result = extract_report(tmp_path)
                        os.unlink(tmp_path)
                        st.write(f"   ✅ Extracted {len(st.session_state.report_result.get('diseases',[]))} diseases, "
                                 f"{len(st.session_state.report_result.get('medications',[]))} medications")
                    except Exception as e:
                        st.warning(f"   ⚠️ Report reader: {e}")
                        st.session_state.report_result = {
                            "diseases": [], "medications": [],
                            "lab_values": {}, "summary": "Could not parse PDF."
                        }

                # Step 3 — XGBoost Risk
                st.write("🤖 Step 2/4 — Predicting risk with XGBoost…")
                try:
                    from risk_predictor import predict_risk
                    vision = st.session_state.vision_result or {}
                    st.session_state.risk_result = predict_risk(
                        st.session_state.patient_info,
                        disease=vision.get("disease", "Unknown")
                    )
                    st.write(f"   ✅ Risk Score: **{st.session_state.risk_result['risk_score']:.1f}%** "
                             f"— {st.session_state.risk_result['risk_level']}")
                except Exception as e:
                    st.warning(f"   ⚠️ Risk predictor: {e}")
                    st.session_state.risk_result = {
                        "risk_score": 0, "risk_level": "UNKNOWN", "top_factors": []
                    }

                # Step 4 — LLM
                st.write("💬 Step 3/4 — Generating AI doctor explanation…")
                try:
                    from ai_doctor import get_diagnosis
                    
                    # Robust disease extraction
                    vision_disease = (st.session_state.vision_result or {}).get("disease")
                    report_diseases = (st.session_state.report_result or {}).get("diseases", [])
                    report_disease = report_diseases[0] if report_diseases else None
                    
                    disease = vision_disease or report_disease or "Unknown"

                    explanation = get_diagnosis(
                        disease=disease,
                        risk_result=st.session_state.risk_result or {},
                        report_result=st.session_state.report_result or {},
                        patient_info=st.session_state.patient_info,
                        api_key=os.environ.get("NVIDIA_API_KEY", ""),
                    )
                    st.session_state.ai_explanation = explanation
                    st.write("   ✅ AI explanation ready")
                except Exception as e:
                    st.warning(f"   ⚠️ AI doctor encountered an error: {e}")
                    # Use rule-based fallback if the engine fails completely
                    try:
                        from ai_doctor import _rule_based_response
                        st.session_state.ai_explanation = _rule_based_response(
                            disease="Unknown",
                            risk_result=st.session_state.risk_result or {},
                            patient_info=st.session_state.patient_info
                        )
                    except:
                        st.session_state.ai_explanation = (
                            "AI explanation unavailable. Please add your NVIDIA API key in the sidebar or check logs."
                        )

                st.write("📊 Step 4/4 — Building health dashboard…")
                st.session_state.analysis_done = True
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

            st.success("🎉 Analysis done! View results in the **Diagnosis Results** and **Health Dashboard** tabs.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Diagnosis Results
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.analysis_done:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#6B7A8D'>
            <div style='font-size:3rem'>🔬</div>
            <div style='font-size:1rem; margin-top:0.5rem'>Run analysis in the <b>Upload & Analyse</b> tab first</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Row 1: detection + risk ──
        col_a, col_b = st.columns([3, 2], gap="large")

        with col_a:
            st.markdown("<div class='med-card'><h3>🩻 Detection Result</h3>", unsafe_allow_html=True)

            vr = st.session_state.vision_result
            rr = st.session_state.report_result
            disease = "N/A"
            confidence = 0.0

            if vr:
                disease    = vr.get("disease", "Unknown")
                confidence = vr.get("confidence", 0.0)
                if vr.get("annotated_image") is not None:
                    st.image(vr["annotated_image"], caption="YOLOv11 Detection Output", use_container_width=True)
                st.markdown(f"""
                <div style='margin-top:0.75rem'>
                    <span style='font-size:1.4rem; font-weight:800; color:#00D4FF'>{disease}</span>
                    <span style='margin-left:1rem; font-family:Space Mono,monospace; color:#00FF88'>{confidence:.1%} confidence</span>
                </div>
                """, unsafe_allow_html=True)

            elif rr:
                diseases = rr.get("diseases", [])
                disease  = diseases[0] if diseases else "See report summary"
                st.markdown(f"""
                <div style='margin-top:0.5rem'>
                    <div style='font-size:1.3rem; font-weight:800; color:#00D4FF'>Report Summary</div>
                </div>
                <div style='margin-top:0.75rem; color:#E8EDF3; line-height:1.7'>{rr.get('summary','—')}</div>
                """, unsafe_allow_html=True)

                if rr.get("diseases"):
                    st.markdown("**Identified Conditions:**")
                    for d in rr["diseases"]:
                        st.markdown(f"• {d}")
                if rr.get("medications"):
                    st.markdown("**Medications Mentioned:**")
                    for m in rr["medications"]:
                        st.markdown(f"• {m}")
                if rr.get("lab_values"):
                    st.markdown("**Lab Values:**")
                    for k, v in rr["lab_values"].items():
                        st.markdown(f"• **{k}**: {v}")

            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            risk = st.session_state.risk_result or {}
            score = risk.get("risk_score", 0)
            level = risk.get("risk_level", "UNKNOWN")
            level_class = {
                "LOW":"risk-low","MEDIUM":"risk-medium",
                "HIGH":"risk-high","CRITICAL":"risk-critical"
            }.get(level, "risk-medium")

            st.markdown(f"""
            <div class='med-card'>
                <h3>⚡ Risk Assessment</h3>
                <div style='text-align:center; padding:1rem 0'>
                    <div style='font-size:3.5rem; font-weight:800; font-family:Space Mono,monospace; color:#00D4FF'>{score:.0f}<span style='font-size:1.5rem'>%</span></div>
                    <div style='margin-top:0.5rem'><span class='risk-badge {level_class}'>{level} RISK</span></div>
                </div>
            """, unsafe_allow_html=True)

            # Progress bar
            bar_color = {"LOW":"#00FF88","MEDIUM":"#FFB830","HIGH":"#FF643C","CRITICAL":"#FF4444"}.get(level,"#00D4FF")
            st.markdown(f"""
                <div style='background:#1a1f2a;border-radius:8px;height:10px;overflow:hidden;margin:0.5rem 0'>
                    <div style='width:{score}%;height:100%;background:linear-gradient(90deg,{bar_color}88,{bar_color});border-radius:8px;transition:width 1s'></div>
                </div>
            """, unsafe_allow_html=True)

            if risk.get("top_factors"):
                st.markdown("<div style='margin-top:0.75rem'><b style='font-size:0.8rem;color:#6B7A8D;text-transform:uppercase;letter-spacing:0.1em'>Top Risk Factors</b></div>", unsafe_allow_html=True)
                for factor, val in risk["top_factors"]:
                    pct = min(abs(val) * 100, 100)
                    color = "#FF4444" if val > 0 else "#00FF88"
                    st.markdown(f"""
                    <div style='margin:4px 0'>
                        <div style='display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px'>
                            <span>{factor}</span><span style='color:{color};font-family:Space Mono,monospace'>{val:+.2f}</span>
                        </div>
                        <div style='background:#1a1f2a;border-radius:4px;height:6px'>
                            <div style='width:{pct}%;height:100%;background:{color}88;border-radius:4px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 2: AI explanation ──
        st.markdown("<div class='med-card'><h3>🤖 AI Doctor Explanation</h3>", unsafe_allow_html=True)

        if st.session_state.ai_explanation:
            exp = st.session_state.ai_explanation
            eng_text = exp.get("english", str(exp)) if isinstance(exp, dict) else str(exp)
            urd_text = exp.get("urdu", "") if isinstance(exp, dict) else ""
            next_steps = exp.get("next_steps", []) if isinstance(exp, dict) else []
            medicines  = exp.get("medicines", []) if isinstance(exp, dict) else []
            lifestyle  = exp.get("lifestyle", []) if isinstance(exp, dict) else []

            st.markdown(f"<div class='ai-response'>{eng_text}</div>", unsafe_allow_html=True)

            if urd_text:
                st.markdown("<br>**اردو خلاصہ:**", unsafe_allow_html=True)
                st.markdown(f"<div class='urdu-text'>{urd_text}</div>", unsafe_allow_html=True)

            if next_steps:
                st.markdown("**📋 Recommended Next Steps:**")
                for s in next_steps:
                    st.markdown(f"• {s}")

            if medicines:
                st.markdown("**💊 Medicines to Ask About:**")
                for m in medicines:
                    st.markdown(f"• {m}")

            if lifestyle:
                st.markdown("**🌿 Lifestyle Suggestions:**")
                for l in lifestyle:
                    st.markdown(f"• {l}")
        else:
            st.info("AI explanation will appear here after analysis.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer' style='margin-top:0.5rem'>
            ⚠️ <strong>Medical Disclaimer:</strong> This AI analysis is for educational purposes only and does NOT replace professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Health Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.analysis_done:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#6B7A8D'>
            <div style='font-size:3rem'>📊</div>
            <div style='font-size:1rem; margin-top:0.5rem'>Run analysis first to see your health dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        from dashboard import (
            render_risk_gauge, render_symptom_chart,
            render_disease_chart, render_shap_chart
        )
        risk   = st.session_state.risk_result or {}
        vision = st.session_state.vision_result or {}

        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig_gauge = render_risk_gauge(risk.get("risk_score", 0), risk.get("risk_level","UNKNOWN"))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            pi = st.session_state.patient_info
            fig_sym = render_symptom_chart(pi.get("symptoms", []))
            st.plotly_chart(fig_sym, use_container_width=True)

        col3, col4 = st.columns(2, gap="large")
        with col3:
            fig_disease = render_disease_chart(vision, st.session_state.report_result)
            st.plotly_chart(fig_disease, use_container_width=True)

        with col4:
            factors = risk.get("top_factors", [])
            if factors:
                fig_shap = render_shap_chart(factors)
                st.plotly_chart(fig_shap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Doctor Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='med-card'><h3>💬 Chat with AI Doctor</h3>", unsafe_allow_html=True)
    st.markdown("<div style='color:#6B7A8D; font-size:0.85rem; margin-bottom:1rem'>Ask follow-up questions about your diagnosis, symptoms, or treatment options.</div>", unsafe_allow_html=True)

    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    chat_input = st.chat_input("Ask the AI doctor a question…")
    if chat_input:
        st.session_state.chat_history.append({"role":"user","content":chat_input})
        with st.spinner("AI doctor is thinking…"):
            try:
                from ai_doctor import chat_with_doctor
                context = {
                    "vision":  st.session_state.vision_result,
                    "risk":    st.session_state.risk_result,
                    "patient": st.session_state.patient_info,
                }
                response = chat_with_doctor(
                    question=chat_input,
                    history=st.session_state.chat_history[:-1],
                    context=context,
                    api_key=os.environ.get("NVIDIA_API_KEY",""),
                )
                st.session_state.chat_history.append({"role":"assistant","content":response})
            except Exception as e:
                st.session_state.chat_history.append({
                    "role":"assistant",
                    "content":f"⚠️ Could not reach AI: {e}. Please add your NVIDIA API key."
                })
        st.rerun()
