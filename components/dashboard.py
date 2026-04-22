"""
FluxAI — Phase C: Visual Command Center (Dashboard)
KPI cards, churn distribution, risk segments, and interactive customer table.
"""
import streamlit as st
import pandas as pd
import re
import io
import plotly.graph_objects as go


# Color palette — light theme
COLORS = {
    'critical': '#DC2626',
    'high': '#EA580C',
    'medium': '#D97706',
    'low': '#059669',
    'bg_card': '#FFFFFF',
    'border': '#E2E8F0',
    'text_muted': '#64748B',
}

RISK_COLORS = {
    'Critical': COLORS['critical'],
    'High': COLORS['high'],
    'Medium': COLORS['medium'],
    'Low': COLORS['low'],
}


def render_dashboard(df):
    """Render the main analytics dashboard."""

    # --- Header ---
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <h2 style="margin: 0; font-weight: 700; color: #1E293B;">Command Center</h2>
            <span style="background: #F1F5F9; color: #64748B; padding: 4px 12px; border-radius: 20px;
                font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; border: 1px solid #E2E8F0;">REAL-TIME</span>
        </div>
        <div style="color: #64748B; font-size: 0.75rem; font-weight: 600;
            display: flex; align-items: center; gap: 0.5rem;">
            <span class="status-dot status-online"></span>
            System Optimized
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- KPI Cards ---
    _render_kpi_cards(df)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # --- Charts Row ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        _render_churn_distribution(df)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        _render_risk_segments(df)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # --- Feature Importance (Global) ---
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    _render_global_feature_importance(df)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # --- Interactive Customer Table ---
    _render_customer_table(df)


def _render_kpi_cards(df):
    """Render the 4 KPI metric cards."""
    total_customers = len(df)
    avg_churn_prob = df['Churn_Probability'].mean()
    critical_count = len(df[df['Risk_Level'] == 'Critical'])
    high_risk_count = len(df[df['Risk_Level'].isin(['Critical', 'High'])])

    # Revenue at risk (customers with >50% churn prob × their monthly charges)
    at_risk = df[df['Churn_Probability'] > 50]
    if 'MonthlyCharges' in df.columns:
        revenue_at_risk = at_risk['MonthlyCharges'].sum() * 12  # Annualized
    else:
        revenue_at_risk = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="premium-card">
            <p style="color: #94A3B8; font-size: 0.65rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                Revenue at Risk</p>
            <h2 style="color: #1E293B; margin: 0.2rem 0; font-size: calc(1.4rem + 0.4vw); font-weight: 800;">
                ${revenue_at_risk:,.0f}</h2>
            <p style="color: #64748B; font-size: 0.6rem; margin: 0;">Projected Annualized</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="premium-card">
            <p style="color: #94A3B8; font-size: 0.65rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                Avg Probability</p>
            <h2 style="color: #6366F1; margin: 0.2rem 0; font-size: calc(1.4rem + 0.4vw); font-weight: 800;">
                {avg_churn_prob:.1f}%</h2>
            <p style="color: #64748B; font-size: 0.6rem; margin: 0;">Metric Stability: Normal</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="premium-card">
            <p style="color: #94A3B8; font-size: 0.65rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                Critical Segments</p>
            <h2 style="color: #EF4444; margin: 0.2rem 0; font-size: calc(1.4rem + 0.4vw); font-weight: 800;">
                {critical_count}</h2>
            <p style="color: #64748B; font-size: 0.6rem; margin: 0;">Probability &gt; 75%</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="premium-card">
            <p style="color: #94A3B8; font-size: 0.65rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">
                High Sensitivity</p>
            <h2 style="color: #F59E0B; margin: 0.2rem 0; font-size: calc(1.4rem + 0.4vw); font-weight: 800;">
                {high_risk_count}</h2>
            <p style="color: #64748B; font-size: 0.6rem; margin: 0;">Probability &gt; 50%</p>
        </div>
        """, unsafe_allow_html=True)


def _render_churn_distribution(df):
    """Render churn probability distribution histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['Churn_Probability'],
        nbinsx=30,
        marker=dict(
            color='rgba(99, 102, 241, 0.65)',
            line=dict(color='#6366f1', width=1),
        ),
        hovertemplate='Probability: %{x:.0f}%<br>Count: %{y}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text="Churn Probability Distribution", font=dict(size=16, color='#1E293B')),
        xaxis=dict(title="Churn Probability (%)", color='#64748B', gridcolor='#F1F5F9'),
        yaxis=dict(title="Number of Customers", color='#64748B', gridcolor='#F1F5F9'),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FAFAFA',
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
        bargap=0.05,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_risk_segments(df):
    """Render risk segment donut chart."""
    risk_counts = df['Risk_Level'].value_counts()

    # Ensure all categories present
    for level in ['Low', 'Medium', 'High', 'Critical']:
        if level not in risk_counts:
            risk_counts[level] = 0

    ordered = ['Low', 'Medium', 'High', 'Critical']
    values = [risk_counts.get(level, 0) for level in ordered]
    colors = [RISK_COLORS[level] for level in ordered]

    fig = go.Figure(data=[go.Pie(
        labels=ordered,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        textfont=dict(size=12, color='#334155'),
        hovertemplate='%{label}: %{value} customers (%{percent})<extra></extra>',
    )])

    fig.update_layout(
        title=dict(text="Risk Segments", font=dict(size=16, color='#1E293B')),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        showlegend=True,
        legend=dict(
            font=dict(color='#64748B', size=11),
            bgcolor='rgba(255,255,255,0)',
        ),
        annotations=[dict(
            text=f"<b>{len(df)}</b><br>Total",
            x=0.5, y=0.5,
            font=dict(size=18, color='#1E293B'),
            showarrow=False,
        )]
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_global_feature_importance(df):
    """Render global SHAP-based feature importance."""
    from utils.predictor import get_predictor

    try:
        predictor = get_predictor()
        importance = predictor.get_global_shap(df, max_samples=min(200, len(df)))

        if importance:
            top_n = importance[:12]  # Top 12
            features = [x[0] for x in top_n][::-1]
            values = [x[1] for x in top_n][::-1]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=features,
                x=values,
                orientation='h',
                marker=dict(
                    color=values,
                    colorscale=[[0, '#059669'], [0.5, '#D97706'], [1, '#DC2626']],
                    line=dict(width=0),
                ),
                hovertemplate='%{y}: %{x:.4f}<extra></extra>',
            ))

            fig.update_layout(
                title=dict(text="Primary Feature Contributions (SHAP)", font=dict(size=14, color='#1E293B')),
                xaxis=dict(title="Mean |SHAP Value|", color='#64748B', gridcolor='#F1F5F9', tickfont=dict(size=10)),
                yaxis=dict(color='#334155', tickfont=dict(size=10)),
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FAFAFA',
                margin=dict(l=140, r=20, t=50, b=40),
                height=380,
            )

            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Global feature importance unavailable: {e}")


def _render_customer_table(df):
    """Render the interactive, searchable customer table."""
    st.markdown("### Risk Registry")

    # Search bar
    search = st.text_input("Search Registry", placeholder="ID, contract, service...", key="customer_search")

    # Filter by risk level
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        options=['Critical', 'High', 'Medium', 'Low'],
        default=['Critical', 'High', 'Medium', 'Low'],
        key="risk_filter",
    )

    # Apply filters
    filtered = df[df['Risk_Level'].isin(risk_filter)]
    if search:
        escaped = re.escape(search)
        mask = filtered.astype(str).apply(
            lambda row: row.str.contains(escaped, case=False, na=False).any(), axis=1
        )
        filtered = filtered[mask]

    # Sort by risk
    filtered = filtered.sort_values('Churn_Probability', ascending=False)

    # Display columns
    display_cols = ['customerID', 'Churn_Probability', 'Risk_Level', 'tenure',
                    'MonthlyCharges', 'Contract', 'InternetService', 'TechSupport']
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
        column_config={
            'customerID': st.column_config.TextColumn('Customer ID', width='medium'),
            'Churn_Probability': st.column_config.ProgressColumn(
                'Churn Risk %',
                format='%.1f%%',
                min_value=0,
                max_value=100,
            ),
            'Risk_Level': st.column_config.TextColumn('Risk Level', width='small'),
            'tenure': st.column_config.NumberColumn('Tenure (months)', width='small'),
            'MonthlyCharges': st.column_config.NumberColumn('Monthly ($)', format='$%.2f'),
            'Contract': st.column_config.TextColumn('Contract'),
            'InternetService': st.column_config.TextColumn('Internet'),
            'TechSupport': st.column_config.TextColumn('Tech Support'),
        },
    )

    st.caption(f"Showing {len(filtered)} of {len(df)} customers")

    # --- Export Results ---
    csv = filtered.to_csv(index=False).encode('utf-8')

    # Also prepare Excel if possible (using openpyxl which is in requirements)
    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            filtered.to_excel(writer, index=False, sheet_name='Churn Risk Analysis')
        excel_data = excel_buffer.getvalue()
        has_excel = True
    except Exception:
        has_excel = False

    export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
    with export_col1:
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name="FluxAI_Analysis_Results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if has_excel:
        with export_col2:
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name="FluxAI_Analysis_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    return filtered
