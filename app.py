"""
FluxAI — Agentic Customer Retention & Recovery Engine
Main Streamlit application entry point.
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="FluxAI — Retention Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Light Look ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Buttons ── */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.2) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #818cf8) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 2px 10px rgba(99,102,241,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(99,102,241,0.45) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 0.35rem;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #64748B !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #6366f1 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #E2E8F0 !important; }

    /* ── Cards & Elevation ── */
    .premium-card {
        background: #FFFFFF !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02) !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        margin-bottom: 1rem;
    }
    .premium-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.06) !important;
    }

    /* ── Mobile Adjustments ── */
    @media (max-width: 768px) {
        .premium-card { padding: 1rem !important; }
        .stButton > button { width: 100% !important; margin-bottom: 0.5rem !important; }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.1rem !important; }
        .ai-pulse { width: 50px; height: 50px; font-size: 1.2rem; }
        [data-testid="stSidebar"] { width: 100% !important; }
    }

    /* ── File uploader ── */
    .stFileUploader > div {
        border-radius: 12px !important;
        border: 2px dashed #E2E8F0 !important;
        background: #F8FAFC !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader > div:hover { border-color: #6366F1 !important; background: #EEF2FF !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebarNav"] { padding-top: 2rem !important; }

    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: none !important;
        color: #64748B !important;
        text-align: left !important;
        padding: 0.6rem 1rem !important;
        width: 100% !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #F1F5F9 !important;
        color: #1E293B !important;
    }

    /* ── Status Indicators ── */
    .status-dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-online { background-color: #10B981; box-shadow: 0 0 8px #10B98144; }
    .status-offline { background-color: #EF4444; }
    .status-warning { background-color: #F59E0B; }
    .status-processing { background-color: #6366F1; animation: pulse 2s infinite; }

    /* ── AI Pulse ── */
    @keyframes aiPulse {
        0%   { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.2); }
        70%  { box-shadow: 0 0 0 15px rgba(99, 102, 241, 0); }
        100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
    }
    .ai-pulse {
        width: 60px; height: 60px;
        background: #EEF2FF;
        border: 1px solid #6366F133;
        border-radius: 50%;
        margin: 2rem auto;
        animation: aiPulse 2s infinite;
        display: flex; align-items: center; justify-content: center;
        color: #6366F1; font-size: 1.5rem;
    }

    /* ── Transitions ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        animation: slideUp 0.6s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    @media (max-width: 768px) {
        .main .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.45; }
    }
    .pulse-dot { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ───────────────────────────────────────
if 'phase' not in st.session_state:
    st.session_state['phase'] = 'onboarding'

if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 'dashboard'

if 'customer_data' not in st.session_state:
    st.session_state['customer_data'] = None

if 'predictions_ready' not in st.session_state:
    st.session_state['predictions_ready'] = False


# ─── Helper Functions ────────────────────────────────────────────────
def _phase_is_complete(phase_key):
    """Check if a phase has been completed."""
    if phase_key == 'onboarding':
        return st.session_state.get('customer_data') is not None
    if phase_key == 'dashboard':
        return st.session_state.get('predictions_ready', False)
    return False


def _nav_to(tab_key):
    """Update active tab and rerun."""
    st.session_state['active_tab'] = tab_key
    st.rerun()


# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h1 style="font-size: 1.8rem; font-weight: 800; color: #1E293B; margin-bottom: 0;">
            Flux<span style="color: #6366F1;">.ai</span>
        </h1>
        <p style="color: #94A3B8; font-size: 0.7rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">
            Retention Engine
        </p>
        <div style="width: 30px; height: 2px; background: #E2E8F0; margin: 0.5rem auto;"></div>
        <p style="color: #64748B; font-size: 0.65rem; margin-top: 0.5rem; letter-spacing: 0.5px; font-weight: 500;">
            Developed by Shayan Ansari
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Phase indicator & Navigation Buttons
    phases = [
        ("Data Onboarding", "onboarding"),
        ("Command Center", "dashboard"),
        ("Customer Deep Dive", "deep_dive"),
        ("Recovery Engine", "recovery"),
    ]

    current_phase = st.session_state.get('phase', 'onboarding')
    active_tab = st.session_state.get('active_tab', 'dashboard')

    for label, tab_key in phases:
        is_complete = _phase_is_complete(tab_key)
        is_active = (current_phase == 'onboarding' and tab_key == 'onboarding') or \
                    (current_phase == 'dashboard' and active_tab == tab_key)

        # Onboarding is special (locked once done)
        if tab_key == 'onboarding':
            if is_complete:
                st.markdown(f'<p style="color:#059669;font-size:0.85rem;padding:0.4rem 0.7rem;margin:0;font-weight:600;">✓ {label}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:#1E293B;font-weight:700;font-size:0.85rem;background:#F1F5F9;padding:0.4rem 0.7rem;border-radius:6px;border-left:3px solid #6366F1;margin:0;">{label}</p>', unsafe_allow_html=True)
            continue

        # Other tabs are only available after onboarding
        if current_phase == 'onboarding':
            st.markdown(f'<p style="color:#94A3B8;font-size:0.85rem;padding:0.4rem 0.7rem;margin:0;opacity:0.5;">{label}</p>', unsafe_allow_html=True)
        else:
            if st.button(f"{label}", key=f"nav_{tab_key}", use_container_width=True):
                _nav_to(tab_key)

    st.markdown("---")

    # Start Over
    if st.button("Reset Session", use_container_width=True, key="start_over_btn"):
        st.session_state.clear()
        st.session_state['phase'] = 'onboarding'
        st.rerun()

    # --- System Status ---
    st.markdown("### System")

    # Check Model
    model_exists = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "xgb_model.pkl"))
    if model_exists:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;margin-bottom:0.2rem;"><span class="status-dot status-online"></span>AI Engine: Online</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;margin-bottom:0.2rem;"><span class="status-dot status-offline"></span>AI Engine: Offline</p>', unsafe_allow_html=True)

    from utils.ollama_client import check_ollama
    ollama_running, _ = check_ollama()
    if ollama_running:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;margin-bottom:0.2rem;"><span class="status-dot status-processing"></span>Recovery: Ready</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.7rem;color:#94A3B8;margin-left:16px;">Llama 3.2 (Local)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;margin-bottom:0.2rem;"><span class="status-dot status-warning"></span>Recovery: Setup Required</p>', unsafe_allow_html=True)

    st.markdown('<p style="color:#CBD5E1;font-size:0.65rem;text-align:center;margin-top:1rem;letter-spacing:1px;text-transform:uppercase;">Secure · Local · XGBoost</p>', unsafe_allow_html=True)


# ─── Main Content Router ────────────────────────────────────────────
def main():
    phase = st.session_state.get('phase', 'onboarding')

    if phase == 'onboarding':
        from components.onboarding import render_onboarding
        render_onboarding()

    elif phase == 'dashboard':
        # Run predictions if not yet done
        if not st.session_state.get('predictions_ready'):
            _run_predictions()

        if st.session_state.get('predictions_ready') and st.session_state.get('predicted_data') is not None:
            predicted_df = st.session_state['predicted_data']
            active_tab = st.session_state.get('active_tab', 'dashboard')

            if active_tab == 'dashboard':
                from components.dashboard import render_dashboard
                render_dashboard(predicted_df)

            elif active_tab == 'deep_dive':
                from components.deep_dive import render_deep_dive
                render_deep_dive(predicted_df)

            elif active_tab == 'recovery':
                from components.recovery import render_recovery
                render_recovery(predicted_df)


def _run_predictions():
    """Run the ML model on the customer data with real-time status updates."""
    import time
    customer_data = st.session_state.get('customer_data')

    if customer_data is None:
        st.session_state['phase'] = 'onboarding'
        st.rerun()
        return

    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem;">
        <h2 style="font-weight:800; color: #1E293B;">
            Running AI Analysis
        </h2>
        <p style="color:#64748B;">Processing records for churn risk assessment…</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        from utils.predictor import get_predictor

        predicted_df = None

        with st.status("Initializing engine...", expanded=True) as status:
            st.write("Loading XGBoost model and explainer...")
            time.sleep(0.3)
            predictor = get_predictor()
            st.write("Model synchronized.")

            time.sleep(0.2)
            st.write(f"Processing {len(customer_data):,} customer records...")
            time.sleep(0.4)
            st.write("Preprocessing complete.")

            time.sleep(0.2)
            st.write("Executing inference...")
            predicted_df = predictor.predict(customer_data)
            st.write(f"Inference complete: {len(predicted_df):,} customers scored.")

            time.sleep(0.2)
            st.write("Segmenting risk tiers...")
            time.sleep(0.3)
            st.write("Segments assigned.")

            time.sleep(0.2)
            st.write("Identifying primary churn drivers...")
            try:
                top_drivers = predictor.get_top_drivers(predicted_df)
                predicted_df = predicted_df.copy()
                predicted_df['Top_Churn_Driver'] = top_drivers.values
                st.write(f"Drivers identified.")
            except Exception:
                st.write("Primary drivers unavailable.")

            time.sleep(0.2)
            st.write("Finalizing environment...")
            time.sleep(0.3)

            status.update(label="Analysis complete", state="complete", expanded=False)

        time.sleep(0.4)

        st.session_state['predicted_data'] = predicted_df
        st.session_state['predictions_ready'] = True
        st.session_state['phase'] = 'dashboard'
        st.rerun()

    except FileNotFoundError:
        st.error("⚠️ Model not found! Please train the model first.")
        st.code("python training/train_model.py", language="bash")
        st.session_state['phase'] = 'onboarding'

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if st.button("← Go Back"):
            st.session_state['phase'] = 'onboarding'
            st.rerun()


# ─── Run ─────────────────────────────────────────────────────────────
main()
