"""
FluxAI — Phase C: Customer Deep Dive Panel (Hardened)
Individual customer profile view with SHAP waterfall/bar chart.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.predictor import get_predictor


def render_deep_dive(df):
    """Render the customer deep-dive panel with SHAP explanations."""

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: clamp(1rem, 3vw, 2rem);">
        <h2 style="margin: 0; font-weight: 700; color: #1E293B; font-size: clamp(1.4rem, 4vw, 1.8rem);">Customer Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    if df is None or len(df) == 0:
        st.warning("No customer data available.")
        return

    df = df.reset_index(drop=True)

    if 'customerID' in df.columns:
        customer_ids = df['customerID'].tolist()
    else:
        customer_ids = [f"Row {i}" for i in range(len(df))]

    if 'Churn_Probability' in df.columns and len(df) > 0:
        default_idx = int(df['Churn_Probability'].idxmax())
    else:
        default_idx = 0
    default_idx = min(default_idx, len(customer_ids) - 1)

    selected_id = st.selectbox("Select a customer to analyze", options=customer_ids, index=default_idx, key="customer_selector")

    if 'customerID' in df.columns:
        matches = df.index[df['customerID'] == selected_id].tolist()
        idx = matches[0] if matches else 0
    else:
        idx = customer_ids.index(selected_id) if selected_id in customer_ids else 0

    customer = df.iloc[idx]
    _render_profile_card(customer)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.session_state['selected_customer_idx'] = idx
    st.session_state['selected_customer'] = customer.to_dict()

    _render_shap_chart(df, idx, customer)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    _render_what_if_simulator(customer)


def _render_profile_card(customer):
    """Render a detailed customer profile card."""
    churn_prob = customer.get('Churn_Probability', 0)
    risk_level = customer.get('Risk_Level', 'Unknown')
    risk_colors = {'Critical': '#DC2626', 'High': '#EA580C', 'Medium': '#D97706', 'Low': '#059669'}
    risk_color = risk_colors.get(str(risk_level), '#64748B')

    try:
        monthly = float(customer.get('MonthlyCharges', 0))
    except (ValueError, TypeError):
        monthly = 0.0
    try:
        total = float(customer.get('TotalCharges', 0))
    except (ValueError, TypeError):
        total = 0.0
    try:
        tenure_val = int(float(customer.get('tenure', 0)))
    except (ValueError, TypeError):
        tenure_val = 0

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style="background: #FFFFFF; padding: 1.2rem; border-radius: 12px; border: 1px solid #E2E8F0;
            text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.02); margin-bottom: 1rem;">
            <p style="color: #94A3B8; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 0.5rem;">
                Customer Identifier</p>
            <h3 style="color: #1E293B; margin: 0 0 0.8rem 0; font-size: 1.1rem; font-weight: 800;">{customer.get('customerID', 'N/A')}</h3>
            <div style="background: {risk_color}10; color: {risk_color}; padding: 10px 20px;
                border-radius: 8px; font-weight: 800; font-size: 1.6rem; display: inline-block; margin-bottom: 0.8rem; border: 1px solid {risk_color}22;">
                {churn_prob:.1f}%</div>
            <p style="color: {risk_color}; font-size: 0.7rem; font-weight: 700; margin: 0; text-transform: uppercase; letter-spacing: 1px;">{risk_level} Segment</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        metrics_row1 = {'Tenure': f"{tenure_val} months", 'Monthly': f"${monthly:.2f}",
                        'Total': f"${total:,.2f}", 'Contract': str(customer.get('Contract', 'N/A'))}
        metrics_row2 = {'Internet': str(customer.get('InternetService', 'N/A')),
                        'Tech Support': str(customer.get('TechSupport', 'N/A')),
                        'Security': str(customer.get('OnlineSecurity', 'N/A')),
                        'Backup': str(customer.get('OnlineBackup', 'N/A'))}

        for metrics_row in [metrics_row1, metrics_row2]:
            cols = st.columns(4)
            for i, (label, value) in enumerate(metrics_row.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: #F8FAFC; padding: 0.6rem 0.8rem; border-radius: 10px;
                        border: 1px solid #E2E8F0; text-align: center;">
                        <span style="color: #94A3B8; font-size: 0.7rem;">{label}</span><br>
                        <span style="color: #1E293B; font-weight: 600; font-size: 0.85rem;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)


def _render_shap_chart(df, idx, customer):
    """Render SHAP explanation bar chart for the selected customer."""
    st.markdown("### Decision Influence Factors")
    try:
        predictor = get_predictor()
        explanation = predictor.explain(df, idx)

        if explanation and explanation.get('feature_names') and explanation.get('shap_values'):
            features = explanation['feature_names']
            values = explanation['shap_values']
            min_len = min(len(features), len(values))
            features, values = features[:min_len], values[:min_len]

            paired = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
            top_n = paired[:12]
            feat_names = [x[0] for x in top_n][::-1]
            shap_vals = [x[1] for x in top_n][::-1]
            colors = ['#DC2626' if v > 0 else '#059669' for v in shap_vals]

            fig = go.Figure()
            fig.add_trace(go.Bar(y=feat_names, x=shap_vals, orientation='h',
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<br><extra></extra>'))

            fig.update_layout(
                title=dict(text="Feature Impact on Churn Prediction", font=dict(size=14, color='#1E293B')),
                xaxis=dict(title="← Reduces Risk | Increases Risk →", color='#64748B',
                    gridcolor='#F1F5F9', zeroline=True, zerolinecolor='#CBD5E1', zerolinewidth=2),
                yaxis=dict(color='#334155'),
                paper_bgcolor='#FFFFFF', plot_bgcolor='#FAFAFA',
                margin=dict(l=160, r=30, t=50, b=50), height=420)

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style="display: flex; gap: 2rem; justify-content: center; margin-top: -0.5rem;">
                <span style="color: #DC2626;">🔴 Increases churn risk</span>
                <span style="color: #059669;">🟢 Decreases churn risk</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("SHAP explanation not available for this customer.")
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")
        st.info("Make sure the model has been trained by running `python training/train_model.py`")


def _render_what_if_simulator(customer):
    """Render an interactive simulator to see how changes affect risk."""
    st.markdown("---")
    with st.expander("Interactive Simulation", expanded=False):
        st.markdown("""<p style="color: #64748B; font-size: 0.9rem; margin-bottom: 1.5rem;">
            Experiment with changing customer attributes to see how it might reduce their churn risk.
            This uses the live AI model to project new probabilities.</p>""", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            c_tenure = st.slider("Tenure (Months)", 0, 72, int(customer.get('tenure', 0)), key="sim_tenure")
            c_monthly = st.slider("Monthly Charges ($)", 0, 200, int(float(customer.get('MonthlyCharges', 0))), key="sim_monthly")
            contracts = ['Month-to-month', 'One year', 'Two year']
            current_contract = customer.get('Contract', 'Month-to-month')
            c_contract = st.selectbox("Contract Type", options=contracts,
                index=contracts.index(current_contract) if current_contract in contracts else 0, key="sim_contract")
            tech_support_opts = ['No', 'Yes', 'No internet service']
            current_tech = customer.get('TechSupport', 'No')
            c_tech = st.selectbox("Tech Support", options=tech_support_opts,
                index=tech_support_opts.index(current_tech) if current_tech in tech_support_opts else 0, key="sim_tech")

        sim_data = customer.to_dict()
        sim_data['tenure'] = c_tenure
        sim_data['MonthlyCharges'] = float(c_monthly)
        sim_data['Contract'] = c_contract
        sim_data['TechSupport'] = c_tech

        try:
            predictor = get_predictor()
            sim_df = pd.DataFrame([sim_data])
            res_df = predictor.predict(sim_df)
            new_prob = res_df.iloc[0]['Churn_Probability']
            old_prob = customer.get('Churn_Probability', 0)
            delta = new_prob - old_prob
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

        with col2:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            color = "#059669" if delta <= 0 else "#DC2626"
            arrow = "↓" if delta <= 0 else "↑"
            st.markdown(f"""
            <div style="background: #FFFFFF; padding: 1.5rem; border-radius: 16px;
                border: 1px solid {color}33; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                <p style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Projected Risk</p>
                <h2 style="color: {color}; margin: 0; font-size: 2.5rem; font-weight: 800;">{new_prob:.1f}%</h2>
                <p style="color: {color}; font-weight: 600; margin-top: 0.5rem;">{arrow} {abs(delta):.1f}% change</p>
            </div>""", unsafe_allow_html=True)
            if delta < -5:
                st.success("✅ Significant risk reduction identified!")
            elif delta > 5:
                st.warning("⚠️ This change increases churn risk.")
