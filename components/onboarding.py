"""
FluxAI — Phase A: Data Onboarding Component
Handles file upload, paste-from-text, template download, and demo mode.
"""
import streamlit as st
import pandas as pd
import io
from utils.data_utils import generate_template, validate_upload, load_sample_data


def render_onboarding():
    """Render the data onboarding interface."""

    # ── Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: clamp(2rem, 5vw, 2.8rem); font-weight: 800; color: #1E293B; margin-bottom: 0.4rem;">
            Flux<span style="color: #6366F1;">.ai</span>
        </h1>
        <p style="font-size: clamp(0.8rem, 2vw, 1rem); color: #64748B; font-weight: 500; margin: 0; letter-spacing: 1px; text-transform: uppercase;">
            Enterprise Retention Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Launch Demo Mode ---
    st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; 
            padding: 1.5rem; text-align: center; margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.03);">
            <div style="max-width: 600px; margin: 0 auto;">
                <h3 style="font-size: clamp(1.1rem, 3vw, 1.25rem); color: #1E293B; font-weight: 700; margin-bottom: 0.8rem;">
                    Launch Interactive Demo
                </h3>
                <p style="color: #64748B; margin-bottom: 1.2rem; font-size: 0.9rem; line-height: 1.5;">
                    Explore Flux.ai instantly with 1,000 pre-loaded customer records. 
                    Test our risk scoring, deep-dive profiles, and recovery playbooks in seconds.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Initialize Demo Workspace", type="primary", use_container_width=True, key="launch_demo_btn"):
        with st.spinner("Loading sample data..."):
            try:
                sample_df = load_sample_data()
                st.session_state['customer_data'] = sample_df
                st.session_state['data_source'] = 'demo'
                st.session_state['phase'] = 'dashboard'
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load sample data: {str(e)}")
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Data Import")

    # ── Tabbed input options ─────────────────────────────────────────────
    tab_upload, tab_paste, tab_template = st.tabs([
        "Upload Dataset",
        "Manual Entry",
        "Download Schema",
    ])

    with tab_upload:
        st.markdown("""
        <p style='color: #64748B; margin-bottom: 0.5rem;'>
            Upload a <strong>CSV</strong> or <strong>Excel (.xlsx)</strong> file containing your customer data.
        </p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your file here or click to browse",
            type=['csv', 'xlsx', 'xls'],
            key='file_uploader',
        )

        if uploaded_file is not None:
            _handle_upload(uploaded_file)

    with tab_paste:
        st.markdown("""
        <p style='color: #64748B; margin-bottom: 0.5rem;'>
            Paste CSV data directly — copy from Excel, Google Sheets, Notepad, or any source.
            The first row should be column headers.
        </p>
        """, unsafe_allow_html=True)

        pasted = st.text_area(
            "Paste your CSV data here",
            height=220,
            placeholder=(
                "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,...\n"
                "A1234,Male,0,Yes,No,12,Yes,...\n"
                "B5678,Female,1,No,No,3,No,..."
            ),
            key="paste_input",
        )

        col_parse, col_clear = st.columns([1, 1])
        with col_parse:
            if st.button("✅ Parse & Preview", type="primary", use_container_width=True, key="parse_btn"):
                if pasted and pasted.strip():
                    _handle_paste(pasted)
                else:
                    st.warning("⚠️ Please paste some data first.")
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_paste_btn"):
                st.session_state.pop('paste_input', None)
                st.session_state.pop('upload_preview', None)
                st.session_state.pop('upload_warnings', None)
                st.rerun()

    with tab_template:
        st.markdown("""
        <p style='color: #64748B; margin-bottom: 1rem;'>
            Download our industry-standard template with column definitions and a data dictionary.
            Fill it in and upload via the <strong>Upload File</strong> tab.
        </p>
        """, unsafe_allow_html=True)

        template_col, _ = st.columns([1, 1])
        with template_col:
            template_buffer = generate_template()
            st.download_button(
                label="Download Schema (.xlsx)",
                data=template_buffer,
                file_name="FluxAI_Customer_Template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.markdown("""
        <div style="margin-top: 1rem; padding: 1rem; background: #F8FAFC;
            border-radius: 10px; border: 1px solid #E2E8F0;">
            <p style="color: #64748B; font-size: 0.85rem; margin: 0;">
                <strong style="color: #1E293B;">Required columns:</strong>
                customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
                PhoneService, MultipleLines, InternetService, OnlineSecurity,
                OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                MonthlyCharges, TotalCharges
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Data Preview (shared across all input methods) ───────────────────
    if 'upload_preview' in st.session_state:
        st.markdown("---")

        preview_df = st.session_state['upload_preview']

        if 'upload_warnings' in st.session_state:
            for warn in st.session_state['upload_warnings']:
                st.warning(warn)

        # ── Action buttons FIRST — fully visible without scrolling ───
        st.markdown(f"**✅ {len(preview_df)} rows detected** — ready to analyse.")
        proceed_col, cancel_col = st.columns([2, 1])
        with proceed_col:
            if st.button("Begin Analysis", type="primary", use_container_width=True, key="proceed_btn"):
                st.session_state['customer_data'] = preview_df
                st.session_state['data_source'] = 'upload'
                st.session_state['phase'] = 'dashboard'
                st.session_state.pop('upload_preview', None)
                st.session_state.pop('upload_warnings', None)
                st.rerun()
        with cancel_col:
            if st.button("✖ Cancel", use_container_width=True, key="cancel_btn"):
                st.session_state.pop('upload_preview', None)
                st.session_state.pop('upload_warnings', None)
                st.rerun()

        # ── Data preview below ────────────────────────────────────────
        with st.expander("📊 Preview data (first 10 rows)", expanded=True):
            st.dataframe(preview_df.head(10), use_container_width=True, height=260)
            st.caption(f"{len(preview_df)} rows · {len(preview_df.columns)} columns detected")



def _handle_paste(text: str):
    """Parse pasted CSV/TSV text into a DataFrame."""
    try:
        # Try tab-separated first (common when copying from Excel/Sheets)
        first_line = text.strip().split('\n')[0]
        separator = '\t' if '\t' in first_line else ','

        df = pd.read_csv(io.StringIO(text), sep=separator)

        if df is None or len(df) == 0:
            st.error("❌ No data rows found. Make sure the first row contains column headers.")
            return

        # Strip leading/trailing whitespace from all column names
        df.columns = [c.strip() for c in df.columns]

        is_valid, cleaned, errors, warnings = validate_upload(df)

        if not is_valid:
            for err in errors:
                st.error(f"❌ {err}")
            st.info(f"📋 Columns detected: {', '.join(df.columns.tolist())}")
            return

        if warnings:
            st.session_state['upload_warnings'] = warnings

        st.session_state['upload_preview'] = cleaned
        st.rerun()

    except Exception as e:
        st.error(f"❌ Could not parse the pasted data: {str(e)}")
        st.info("Make sure the data is comma or tab separated with a header row.")


def _handle_upload(uploaded_file):
    """Process an uploaded file with robust error handling."""
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("❌ The CSV file is empty — no data to process.")
                return
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        else:
            try:
                df = pd.read_excel(uploaded_file, sheet_name=0)
            except Exception as e:
                st.error(f"❌ Could not read Excel file: {str(e)}")
                return

        if df is None or len(df) == 0:
            st.error("❌ The uploaded file contains no data rows.")
            return

        is_valid, cleaned, errors, warnings = validate_upload(df)

        if not is_valid:
            for err in errors:
                st.error(f"❌ {err}")
            st.info(f"📋 Columns found in your file: {', '.join(df.columns.tolist())}")
            return

        if errors:
            for err in errors:
                st.warning(err)

        if warnings:
            st.session_state['upload_warnings'] = warnings

        st.session_state['upload_preview'] = cleaned
        st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to read file: {str(e)}")
        st.info("Make sure the file is a valid CSV or Excel file with customer data.")
