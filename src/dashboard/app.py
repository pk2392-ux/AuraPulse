import os
import sys
import json
import hashlib
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# --- PATH INJECTION ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.processors.data_loader import run_janitor_purge, load_historical_data, METRIC_CONFIG, STORAGE_DIR
from src.agents.crew_manager import run_health_analysis, generate_chat_response

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AuraPulse AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "collapse_sidebar" not in st.session_state:
    st.session_state["collapse_sidebar"] = False

if "ai_briefing" not in st.session_state:
    st.session_state["ai_briefing"] = None

# --- CUSTOM CSS (MODERN GLASSMORPHISM HUD) ---
st.markdown("""
<style>
    /* Global Background & Font - Deep Space Elegant */
    .stApp {
        background-color: #030712;
        background-image: radial-gradient(circle at 50% 0%, #111827 0%, #030712 70%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, sans-serif;
    }

    /* Clean headers */
    h1, h2, h3, h4, h5 {
        color: #f8fafc !important;
        font-weight: 300;
        letter-spacing: 2px;
    }

    /* ⚡ AURAPULSE NEON LOGO HEADER ⚡ */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@900&display=swap');

    .aurapulse-logo {
        text-align: left;
        margin-top: -30px;
        margin-bottom: 30px;
        padding-bottom: 20px;
        padding-left: 10px;
    }
    .aurapulse-text {
        font-size: 42px;
        font-weight: 900;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 6px;
        color: #fff;
        /* Softer Neon Tube Effect */
        text-shadow: 
            0 0 2px rgba(255, 255, 255, 0.8),
            0 0 8px rgba(0, 240, 255, 0.6),
            0 0 15px rgba(0, 240, 255, 0.4),
            0 0 25px rgba(0, 240, 255, 0.2);
    }
    .aurapulse-subtext {
        font-size: 11px;
        color: #00f0ff; 
        letter-spacing: 5px;
        text-transform: uppercase;
        margin-top: 5px;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 3px rgba(0, 240, 255, 0.3); /* Softer subtext glow */
    }

    /* Elegant Glassmorphism Containers */
    .hud-container {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .hud-container:hover {
        border: 1px solid rgba(0, 240, 255, 0.3);
        box-shadow: 0 4px 30px rgba(0, 240, 255, 0.05);
    }

    /* Section Subheaders */
    .section-header {
        font-size: 14px;
        color: #00f0ff;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    .section-header::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #00f0ff;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 0 8px #00f0ff;
    }

    /* Modern Progress Bars */
    .metric-row { margin-bottom: 22px; }
    .metric-labels {
        display: flex; justify-content: space-between; font-size: 13px;
        color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-value-text {
        color: #ffffff; font-weight: 600; font-size: 15px; text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
    .progress-bg {
        background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; height: 6px; width: 100%; position: relative; overflow: hidden;
    }
    .progress-fill {
        height: 100%; position: absolute; top: 0; left: 0; border-radius: 10px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .fill-mint { background-color: #00f0ff; box-shadow: 0 0 10px #00f0ff; }
    .fill-amber { background-color: #f59e0b; box-shadow: 0 0 10px #f59e0b; }
    .fill-blue { background-color: #3b82f6; box-shadow: 0 0 10px #3b82f6; }

    /* Elegant Status Panel */
    .status-panel {
        display: flex; gap: 20px; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .status-item {
        font-size: 11px; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; color: #cbd5e1;
    }
    .status-dot { width: 6px; height: 6px; border-radius: 50%; margin-right: 8px; }
    .dot-stable { background-color: #00f0ff; box-shadow: 0 0 8px #00f0ff; }
    .dot-warn { background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
    .dot-alert { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }

    /* AI Briefing Text */
    .ai-briefing { font-size: 15px; line-height: 1.7; color: #e2e8f0; font-weight: 300; }
    .ai-briefing strong { color: #00f0ff; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }

    /* Plotly Overrides */
    div[data-testid="stPlotlyChart"] { background: transparent !important; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(3, 7, 18, 0.95) !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        background-color: rgba(255, 255, 255, 0.03) !important; color: #00f0ff !important; border: 1px solid rgba(0, 240, 255, 0.3) !important;
        border-radius: 8px; text-transform: uppercase; letter-spacing: 1px; font-size: 12px; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: rgba(0, 240, 255, 0.1) !important; border-color: #00f0ff !important; box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
    }
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(0, 240, 255, 0.3); background: rgba(255, 255, 255, 0.02); border-radius: 12px; padding: 15px; transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border: 1px dashed rgba(0, 240, 255, 0.8); background: rgba(0, 240, 255, 0.05);
    }

    /* Chat bot custom style */
    .stChatMessage {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(0, 240, 255, 0.1) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

files_purged = run_janitor_purge(STORAGE_DIR, days_threshold=30)


@st.cache_data(ttl=60)
def get_cached_historical_data(directory):
    return load_historical_data(directory)


# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("<h3 style='color: #00f0ff !important; font-size: 16px; margin-bottom: 20px;'>AURA // UPLINK</h3>",
                unsafe_allow_html=True)
    if st.session_state.get("show_success"):
        st.success("SYNC SUCCESSFUL.")
        st.session_state["show_success"] = False

    uploaded_file = st.file_uploader(label="Sync Telemetry Payload", type=["json"],
                                     key=f"uploader_{st.session_state['uploader_key']}", label_visibility="collapsed")
    st.caption("Drop health telemetry (.json)")

    st.markdown("<br><h4 style='color: #8ab4f8; font-size: 14px;'>BIO-PROFILE</h4>", unsafe_allow_html=True)
    goal = st.selectbox("Primary Goal",
                        ["General Health", "Weight Loss", "Muscle Gain", "Athletic Performance", "Longevity"])
    activity_level = st.selectbox("Baseline Activity", ["Sedentary", "Moderate", "Active", "Highly Active"])
    caffeine = st.selectbox("Caffeine Intake",
                            ["None", "Low (1-2 cups)", "Moderate (3-4 cups)", "High (5+ cups)", "Late Day"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Rarely", "Weekends", "Daily"])
    late_meals = st.selectbox("Late Meals (Within 2 hrs of sleep)", ["Rarely", "Sometimes", "Often"])

    # We store the user profile in session state so the chat bot can access it too!
    st.session_state["user_profile"] = {
        "primary_goal": goal,
        "activity_level": activity_level,
        "caffeine": caffeine,
        "alcohol": alcohol,
        "late_meals": late_meals
    }

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        success = False
        if "last_processed_hash" not in st.session_state or st.session_state.last_processed_hash != file_hash:
            try:
                file_payload = json.loads(file_bytes)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_filename = f"apple_health_{timestamp}.json"
                target_path = os.path.join(STORAGE_DIR, target_filename)
                with open(target_path, "w") as f:
                    json.dump(file_payload, f, indent=4)
                st.session_state.last_processed_hash = file_hash
                get_cached_historical_data.clear()
                st.session_state["ai_briefing"] = None  # Reset briefing so it generates a new one
                st.session_state["uploader_key"] += 1
                success = True
            except Exception as e:
                st.error("SYNC FAILED.")
        if success:
            st.session_state["show_success"] = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    vault_files = [f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')]
    st.markdown(
        f"<span style='color: #94a3b8; font-size: 12px;'>DATABANKS:</span> <strong style='color:#00f0ff;'>{len(vault_files)} ACTIVE</strong>",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("PURGE DATABANKS", width="stretch"):
        for f in os.listdir(STORAGE_DIR):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(STORAGE_DIR, f))
                except:
                    pass
        get_cached_historical_data.clear()
        st.session_state["ai_briefing"] = None
        st.session_state["uploader_key"] += 1
        if "last_processed_hash" in st.session_state: del st.session_state["last_processed_hash"]
        st.rerun()

df_historical = get_cached_historical_data(STORAGE_DIR)

if not df_historical.empty:
    components.html(
        """<script>setTimeout(function() { const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]'); if (sidebar && sidebar.getAttribute('aria-expanded') === 'true') { const btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]'); if (btn) { btn.click(); } } }, 300);</script>""",
        height=0, width=0)

# ─── MAIN DASHBOARD ───
# Custom Animated AuraPulse Logo Header
st.markdown("""
<div class="aurapulse-logo">
    <div class="aurapulse-text">AURAPULSE</div>
    <div class="aurapulse-subtext">Cognitive Health Intelligence</div>
</div>
""", unsafe_allow_html=True)

if not df_historical.empty:
    latest = df_historical.iloc[-1]

    # ─── RUN CREW AI (Briefing) ───
    if st.session_state["ai_briefing"] is None:
        with st.spinner("🧠 E.V.E. is assembling specialist crew and analyzing databanks..."):
            briefing = run_health_analysis(df_historical.to_dict('records'), st.session_state["user_profile"])
            st.session_state["ai_briefing"] = briefing

    # Extract dynamic status lights based on raw data (simple logic for UI)
    hr_val = latest.get("Resting HR", 60)
    slp_val = latest.get("Sleep Duration", 8)
    steps_val = latest.get("Steps", 10000)

    hr_status = '<div class="status-item"><div class="status-dot dot-stable"></div> CARDIAC: OPTIMAL</div>' if hr_val <= 65 else '<div class="status-item"><div class="status-dot dot-warn"></div> CARDIAC: ELEVATED</div>'
    slp_status = '<div class="status-item"><div class="status-dot dot-stable"></div> SLEEP: OPTIMAL</div>' if slp_val >= 7.5 else '<div class="status-item"><div class="status-dot dot-alert"></div> SLEEP: DEFICIT</div>'
    kin_status = '<div class="status-item"><div class="status-dot dot-stable"></div> KINETIC: OPTIMAL</div>' if steps_val >= 8000 else '<div class="status-item"><div class="status-dot dot-warn"></div> KINETIC: DEFICIT</div>'

    st.markdown(f"""
    <div class="hud-container">
        <div class="status-panel">
            {hr_status}
            {slp_status}
            {kin_status}
        </div>
        <div class="ai-briefing">
            {st.session_state['ai_briefing']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("<div class='section-header' style='margin-top: 10px;'>LATEST BIOMETRICS</div>",
                    unsafe_allow_html=True)


        def make_progress_bar(label, value, target, unit, fill_class="fill-blue"):
            if pd.isna(value): return ""
            pct = min(100, max(0, (value / target) * 100))
            return f"""
            <div class="metric-row">
                <div class="metric-labels"><span>{label}</span><span><span class="metric-value-text">{value:,.1f}</span> / {target:,.1f} {unit}</span></div>
                <div class="progress-bg"><div class="progress-fill {fill_class}" style="width: {pct}%;"></div></div>
            </div>
            """


        st.markdown(make_progress_bar("RESTING HR", latest.get("Resting HR", float('nan')), 60, "BPM", "fill-amber"),
                    unsafe_allow_html=True)
        st.markdown(
            make_progress_bar("SLEEP CYCLE", latest.get("Sleep Duration", float('nan')), 8.0, "HRS", "fill-mint"),
            unsafe_allow_html=True)
        st.markdown(make_progress_bar("KINETIC OUTPUT", latest.get("Steps", float('nan')), 10000, "STP", "fill-blue"),
                    unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-header' style='margin-left:10px; margin-top:10px;'>TELEMETRY VECTORS</div>",
                    unsafe_allow_html=True)
        active_metrics = [m for m in METRIC_CONFIG.values() if m["label"] in df_historical.columns]
        grid_cols = st.columns(2)
        for i, metric in enumerate(active_metrics):
            g_col = grid_cols[i % 2]
            with g_col:
                fig = px.area(df_historical, x="Date", y=metric["label"], title=f"{metric['label'].upper()}")
                fig.update_traces(line_color=metric["color"], fillcolor=metric["color"], line_width=2,
                                  marker=dict(size=5, symbol='circle', color='#ffffff',
                                              line=dict(width=1, color=metric["color"])))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#94a3b8", family="Inter, sans-serif", size=10),
                                  title_font=dict(size=12, color="#f8fafc", family="Inter, sans-serif"),
                                  margin=dict(l=0, r=0, t=30, b=0), height=130,
                                  xaxis=dict(showgrid=False, zeroline=False, visible=False),
                                  yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title=""))
                st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
            if i % 2 == 1 and i != len(active_metrics) - 1:
                grid_cols = st.columns(2)

    st.markdown("---")

    # ─── ROW 3: INTERACTIVE AI COMPANION ───
    st.markdown("<div class='section-header'>🤖 E.V.E. COGNITIVE INTERFACE</div>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "assistant",
                                             "content": "SYSTEM ONLINE. I am E.V.E., your Cognitive Health Intelligence. How can I assist you with your telemetry data today?"}]

    # UI DISPLAY LOGIC: Only show the most recent Q&A (last 2 messages) or the initial greeting
    display_messages = st.session_state["chat_history"][-2:] if len(st.session_state["chat_history"]) > 1 else st.session_state["chat_history"]

    for message in display_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_prompt := st.chat_input("Ask E.V.E. about your data..."):
        # We don't render it here immediately, because the page will re-render in a moment anyway,
        # but we MUST add it to the background history so the API sees it.
        st.session_state["chat_history"].append({"role": "user", "content": user_prompt})

        # Display a "thinking" spinner while the API call runs
        with st.chat_message("assistant"):
            with st.spinner("⚙️ E.V.E. is processing..."):
                # We call the AI, passing the FULL history so it has context
                response_text = generate_chat_response(
                    prompt=user_prompt,
                    chat_history=st.session_state["chat_history"],
                    historical_metrics=df_historical.to_dict('records'),
                    user_profile=st.session_state["user_profile"]
                )

        # Save AI response to history
        st.session_state["chat_history"].append({"role": "assistant", "content": response_text})
        st.rerun()

else:
    st.markdown("""
    <div class="hud-container" style="text-align:center; padding: 60px;">
        <h3 style="color: #00f0ff;">SYSTEM STANDBY</h3>
        <p style="color: #94a3b8;">No telemetry detected. Establish uplink via the sidebar terminal.</p>
    </div>
    """, unsafe_allow_html=True)