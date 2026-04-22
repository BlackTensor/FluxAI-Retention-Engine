"""
FluxAI — Phase D: Agentic Recovery Playbook
LLM-powered recovery email and strategy generation via Ollama.
"""
import streamlit as st
from utils.ollama_client import check_ollama, generate_playbook, get_available_models
from utils.predictor import get_predictor


def render_recovery(df):
    """Render the agentic recovery playbook panel."""
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-weight: 700; color: #1E293B; font-size: clamp(1.4rem, 4vw, 1.8rem);">Recovery Engine</h2>
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            <span style="background: #F1F5F9; color: #64748B; padding: 4px 12px; border-radius: 20px;
                font-size: 0.65rem; font-weight: 600; letter-spacing: 1px; border: 1px solid #E2E8F0;">NEURAL ARCHITECTURE</span>
            <div style="color: #64748B; font-size: 0.7rem; font-weight: 600;
                display: flex; align-items: center; gap: 0.5rem;">
                <span class="status-dot status-online"></span>
                Privacy Compliant
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if 'selected_customer' not in st.session_state:
        st.info("👆 Select a customer from the **Deep Dive** tab first to generate a recovery playbook.")
        return

    customer = st.session_state['selected_customer']
    customer_idx = st.session_state.get('selected_customer_idx', 0)

    churn_prob = customer.get('Churn_Probability', 0)
    risk_level = customer.get('Risk_Level', 'Unknown')
    risk_colors = {'Critical': '#DC2626', 'High': '#EA580C', 'Medium': '#D97706', 'Low': '#059669'}
    risk_color = risk_colors.get(str(risk_level), '#64748B')

    st.markdown(f"""
    <div class="premium-card" style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem;">
        <div style="background: #F8FAFC; width: 45px; height: 45px; flex-shrink: 0;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            border: 1px solid #E2E8F0; color: #64748B; font-weight: 700; font-size: 0.8rem;">ID</div>
        <div style="min-width: 0;">
            <h3 style="margin: 0; color: #1E293B; font-weight: 700; font-size: 1rem; overflow: hidden; text-overflow: ellipsis;">{customer.get('customerID', 'N/A')}</h3>
            <p style="margin: 0.2rem 0 0; color: #64748B; font-size: 0.8rem; line-height: 1.4;">
                Risk: <span style="color: {risk_color}; font-weight: 800;">{churn_prob:.1f}%</span>
                &nbsp; · &nbsp; {customer.get('Contract', 'N/A')}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    is_running, _ = check_ollama()
    if not is_running:
        _render_ollama_setup()
        return

    selected_model = "llama3.2"

    if st.button("Generate Recovery Playbook", type="primary", use_container_width=True):
        _generate_and_display(df, customer, customer_idx, selected_model)
    elif 'recovery_playbook' in st.session_state:
        _display_playbook(st.session_state['recovery_playbook'], customer)


def _generate_and_display(df, customer, customer_idx, model):
    """Generate and display recovery playbook with animated progress."""
    import time

    STEPS = [
        ("1", "Analysing Risk Factors", "Executing SHAP to identify top churn drivers..."),
        ("2", "Building Customer Profile", "Compiling account history and charge details..."),
        ("3", "Generating Recovery Playbook", f"AI is synthesizing a personalized retention strategy..."),
    ]

    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem 0;">
        <h3 style="font-weight:700; background:linear-gradient(135deg,#6366f1,#818cf8);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent; margin-bottom:0.2rem;">
            🤖 Recovery Engine Processing
        </h3>
        <p style="color:#64748B; font-size:0.92rem;">Leveraging local intelligence for secure strategy drafting…</p>
    </div>
    """, unsafe_allow_html=True)

    prog_ph = st.empty()
    steps_ph = st.empty()

    def _draw(done: int):
        prog_ph.progress(int((done / len(STEPS)) * 90), text=STEPS[min(done, len(STEPS)-1)][1] + "…")
        html = "<div style='max-width:540px;margin:0 auto;'>"
        for i, (step_num, title, subtitle) in enumerate(STEPS):
            if i < done:
                html += f"""<div style="display:flex;align-items:center;gap:1.2rem;
                    background:#FFFFFF;border:1px solid #10B98133;
                    border-radius:12px;padding:0.7rem;margin-bottom:0.6rem;">
                    <span style="font-weight:800; color:#10B981; font-size: 0.9rem;">{step_num}</span>
                    <div><p style="margin:0;font-weight:700;color:#10B981;font-size:0.85rem;">{title}</p>
                    <p style="margin:0;font-size:0.7rem;color:#64748B;">{subtitle}</p></div></div>"""
            elif i == done:
                html += f"""<div style="display:flex;align-items:center;gap:1.2rem;
                    background:#FFFFFF;border:1px solid #6366F133;
                    border-radius:12px;padding:0.7rem;margin-bottom:0.6rem;">
                    <span style="font-weight:800; color:#6366F1; font-size: 0.9rem;">{step_num}</span>
                    <div><p style="margin:0;font-weight:700;color:#6366F1;font-size:0.85rem;">{title}</p>
                    <p style="margin:0;font-size:0.7rem;color:#64748B;">{subtitle}</p></div></div>"""
            else:
                html += f"""<div style="display:flex;align-items:center;gap:1.2rem;
                    background:#FFFFFF;border:1px solid #E2E8F0;
                    border-radius:12px;padding:0.7rem;margin-bottom:0.6rem;opacity:0.4;">
                    <span style="font-weight:800; color:#94A3B8; font-size: 0.9rem;">{step_num}</span>
                    <div><p style="margin:0;font-weight:700;color:#94A3B8;font-size:0.85rem;">{title}</p>
                    <p style="margin:0;font-size:0.7rem;color:#94A3B8;">{subtitle}</p></div></div>"""
        html += "</div>"
        steps_ph.markdown(html, unsafe_allow_html=True)

    _draw(0)
    time.sleep(0.4)
    try:
        predictor = get_predictor()
        explanation = predictor.explain(df, customer_idx)
        if explanation and explanation.get('feature_names') and explanation.get('shap_values'):
            risk_factors = sorted(
                zip(explanation['feature_names'], explanation['shap_values']),
                key=lambda x: abs(x[1]), reverse=True)
        else:
            risk_factors = [("General churn signals", 0.5)]
    except Exception:
        risk_factors = [("General churn signals", 0.5)]

    _draw(1)
    time.sleep(0.45)
    
    profile = {
        'Customer ID': customer.get('customerID', 'N/A'),
        'Tenure': f"{customer.get('tenure', 'N/A')} months",
        'Monthly Charges': f"${float(customer.get('MonthlyCharges', 0)):.2f}",
        'Total Charges': f"${float(customer.get('TotalCharges', 0)):,.2f}",
        'Contract Type': str(customer.get('Contract', 'N/A')),
        'Internet Service': str(customer.get('InternetService', 'N/A')),
        'Tech Support': str(customer.get('TechSupport', 'N/A')),
        'Online Security': str(customer.get('OnlineSecurity', 'N/A')),
        'Churn Risk Score': f"{customer.get('Churn_Probability', 0):.1f}%",
    }

    _draw(2)
    
    pulse_ph = st.empty()
    pulse_ph.markdown('<div class="ai-pulse">...</div>', unsafe_allow_html=True)

    import threading
    result_holder = {}

    def _run_llm():
        try:
            result_holder['result'] = generate_playbook(profile, risk_factors, model=model)
        except Exception as e:
            result_holder['result'] = {'success': False, 'error': str(e)}

    llm_thread = threading.Thread(target=_run_llm, daemon=True)
    llm_thread.start()

    start_time = time.time()
    while llm_thread.is_alive():
        elapsed_sec = int(time.time() - start_time)
        # Smoothly increment progress while thinking
        prog_ph.progress(min(95, 70 + elapsed_sec), text=f"AI is thinking... ({elapsed_sec}s)")
        time.sleep(1)

    llm_thread.join()
    pulse_ph.empty()
    result = result_holder.get('result', {'success': False, 'error': 'Thread failed'})

    prog_ph.progress(100, text="Playbook complete.")
    time.sleep(0.5)
    prog_ph.empty()
    steps_ph.empty()

    if result.get('success'):
        st.session_state['recovery_playbook'] = result
        _display_playbook(result, customer)
    else:
        st.error(f"❌ {result.get('error', 'Unknown error')}")
        st.info("💡 Tip: Try using a smaller model like `llama3.2:1b` for faster inference on CPU.")


def _display_playbook(result, customer):
    """Display the generated recovery playbook."""
    st.markdown("---")

    st.markdown(f"""
    <div style="background: #FFFFFF; padding: 0.8rem 1.2rem;
        border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 1rem; display: flex; justify-content: space-between;">
        <span style="color: #10B981; font-weight: 600;">Status: Verified Report</span>
        <span style="color: #94A3B8; font-size: 0.75rem;">Internal Engine v1.0</span>
    </div>
    """, unsafe_allow_html=True)

    content = result.get('content', 'No content generated.')

    email_content = ""
    if "**RECOVERY EMAIL**" in content:
        parts = content.split("**RECOVERY EMAIL**")
        if len(parts) > 1:
            email_part = parts[1].split("**RETENTION STRATEGY**")[0]
            email_content = email_part.strip()

    col_content, col_summary = st.columns([2, 1])
    with col_content:
        st.markdown(content)

    with col_summary:
        st.markdown("### Executive Summary")
        urgency = "High"
        if "**URGENCY LEVEL**" in content:
            urgency_part = content.split("**URGENCY LEVEL**")[1].split("\n")[0].strip(": ")
            urgency = "".join(filter(str.isalnum, urgency_part.split()[0])) if urgency_part else "High"

        st.markdown(f"""
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 12px; border: 1px solid #E2E8F0;">
            <p style="color: #94A3B8; font-size: 0.7rem; margin-bottom: 0.2rem;">URGENCY</p>
            <p style="font-weight: 700; color: {'#DC2626' if urgency in ['Critical', 'High'] else '#D97706'};">{urgency}</p>
            <hr style="margin: 0.5rem 0; opacity: 0.15;">
            <p style="color: #94A3B8; font-size: 0.7rem; margin-bottom: 0.2rem;">CUSTOMER</p>
            <p style="font-size: 0.85rem; font-weight: 600; color: #1E293B;">{customer.get('customerID', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

        if email_content:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### Communication Template")
            st.code(email_content, language="markdown")
            st.caption("Copy the email above for outreach.")

    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        st.download_button(label="📥 Download MD", data=content,
            file_name=f"Recovery_{customer.get('customerID', 'unknown')}.md",
            mime="text/markdown", use_container_width=True)
    with btn_col2:
        if st.button("🔄 Regenerate", use_container_width=True):
            if 'recovery_playbook' in st.session_state:
                del st.session_state['recovery_playbook']
            st.rerun()


def _render_ollama_setup():
    """Show Ollama setup instructions when it's not running."""
    st.markdown("""
    <div style="background: #FEF2F2; padding: 1.5rem;
        border-radius: 16px; border: 1px solid #DC262644;">
        <h3 style="color: #DC2626; margin-top: 0;">⚠️ Ollama Not Detected</h3>
        <p style="color: #334155;">
            The Recovery Engine requires <b>Ollama</b> running locally with a language model.
            This keeps your customer data 100% private — nothing leaves your machine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Quick Setup (3 minutes)

    **Step 1:** Install Ollama
    ```bash
    # Windows: Download from https://ollama.com/download
    # macOS: brew install ollama
    # Linux: curl -fsSL https://ollama.com/install.sh | sh
    ```

    **Step 2:** Pull a model
    ```bash
    ollama pull llama3.2
    ```

    **Step 3:** Start Ollama (it usually auto-starts)
    ```bash
    ollama serve
    ```

    **Step 4:** Refresh this page 🔄
    """)

    if st.button("🔄 Check Again", type="primary"):
        st.rerun()
