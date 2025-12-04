"""
STREAMLIT DASHBOARD (MVP)

- Load data/gold/anomalies.csv (pandas)
- Show KPIs: total claims, total anomalies, anomaly rate
- Bar chart: anomalies by flag type
- Table: top 20 anomalies including 'explanation' column
- Filters: provider_id, date range
- Auto-runs pipeline on startup if data doesn't exist
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Page configuration
st.set_page_config(
    page_title="Abacus DQ Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def run_pipeline_once():
    """Run the pipeline once on startup if data doesn't exist."""
    data_dir = Path("data/gold")
    anomalies_file = data_dir / "anomalies.csv"
    
    # Check if data exists
    if not anomalies_file.exists():
        with st.spinner("🔄 Running data pipeline for the first time..."):
            try:
                from src.ingestion.ingest import run_ingest
                from src.transform.transform import run_transform
                from src.dq.rules import run_dq
                from src.anomaly.detect import detect_anomalies
                from src.llm.explain import explain_anomaly
                
                # Run pipeline steps
                st.info("Step 1/5: Ingesting data...")
                run_ingest()
                
                st.info("Step 2/5: Transforming data...")
                transform_results = run_transform()
                
                st.info("Step 3/5: Running DQ rules...")
                dq_anomalies = run_dq()
                
                st.info("Step 4/5: Detecting anomalies...")
                anomalies = detect_anomalies(dq_anomalies)
                
                st.info("Step 5/5: Generating explanations...")
                if anomalies is not None and len(anomalies) > 0:
                    # Generate explanations for top 20
                    anomalies_sorted = anomalies.sort_values("num_flags", ascending=False) if "num_flags" in anomalies.columns else anomalies
                    explanations = []
                    for idx, row in anomalies_sorted.head(20).iterrows():
                        record = row.to_dict()
                        explanation = explain_anomaly(record)
                        explanations.append(explanation)
                    
                    anomalies["explanation"] = ""
                    top_indices = anomalies_sorted.head(20).index
                    for i, idx in enumerate(top_indices):
                        anomalies.loc[idx, "explanation"] = explanations[i]
                    
                    # Save outputs
                    output_dir = Path("data/gold")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    anomalies.to_parquet(output_dir / "anomalies.parquet", index=False)
                    anomalies.to_csv(output_dir / "anomalies.csv", index=False)
                
                st.success("✅ Pipeline completed successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                return False
    return True


@st.cache_data
def load_data():
    """Load anomalies and claims data."""
    data_dir = Path("data/gold")
    
    # Load anomalies
    anomalies_file = data_dir / "anomalies.csv"
    if anomalies_file.exists():
        anomalies = pd.read_csv(anomalies_file)
    else:
        anomalies = pd.DataFrame()
    
    # Load clean claims for total count
    clean_file = data_dir / "claims_clean.parquet"
    if clean_file.exists():
        clean_claims = pd.read_parquet(clean_file)
        total_claims = len(clean_claims) + len(anomalies)
    else:
        total_claims = len(anomalies)
    
    return anomalies, total_claims


def get_anomaly_counts(anomalies: pd.DataFrame) -> pd.Series:
    """Count anomalies by flag type."""
    if "anomaly_reasons_str" not in anomalies.columns:
        return pd.Series()
    
    all_reasons = []
    for reasons_str in anomalies["anomaly_reasons_str"].dropna():
        if reasons_str:
            all_reasons.extend(reasons_str.split(","))
    
    return pd.Series(all_reasons).value_counts()


def app():
    """Main Streamlit dashboard application."""
    
    # Auto-run pipeline if data doesn't exist
    run_pipeline_once()
    
    # Header
    st.title("📊 Abacus Data Quality Dashboard")
    st.markdown("Healthcare Claims Anomaly Detection & Analysis")
    
    # Load data
    anomalies, total_claims = load_data()
    
    if anomalies.empty:
        st.error("No anomaly data found. Please run the pipeline first: `python src/run_pipeline.py`")
        st.info("💡 Tip: Add CSV files to data/ folder and refresh the page to auto-run the pipeline.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Provider filter
    if "provider_id" in anomalies.columns:
        providers = ["All"] + sorted(anomalies["provider_id"].dropna().unique().tolist())
        selected_provider = st.sidebar.selectbox("Provider ID", providers)
    else:
        selected_provider = "All"
    
    # Date range filter
    if "service_date" in anomalies.columns:
        anomalies["service_date"] = pd.to_datetime(anomalies["service_date"], errors="coerce")
        valid_dates = anomalies["service_date"].dropna()
        
        if len(valid_dates) > 0:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            date_range = st.sidebar.date_input(
                "Service Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = None
    else:
        date_range = None
    
    # Anomaly type filter
    anomaly_counts = get_anomaly_counts(anomalies)
    if len(anomaly_counts) > 0:
        anomaly_types = ["All"] + anomaly_counts.index.tolist()
        selected_anomaly_type = st.sidebar.selectbox("Anomaly Type", anomaly_types)
    else:
        selected_anomaly_type = "All"
    
    # Apply filters
    filtered = anomalies.copy()
    
    if selected_provider != "All":
        filtered = filtered[filtered["provider_id"] == selected_provider]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (filtered["service_date"].dt.date >= start_date) & (filtered["service_date"].dt.date <= end_date)
        filtered = filtered[mask | filtered["service_date"].isna()]
    
    if selected_anomaly_type != "All":
        filtered = filtered[filtered["anomaly_reasons_str"].str.contains(selected_anomaly_type, na=False)]
    
    # KPI Cards
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Claims",
            value=f"{total_claims:,}"
        )
    
    with col2:
        st.metric(
            label="Total Anomalies",
            value=f"{len(anomalies):,}",
            delta=f"{len(anomalies)/total_claims*100:.1f}%" if total_claims > 0 else "N/A"
        )
    
    with col3:
        st.metric(
            label="Filtered Anomalies",
            value=f"{len(filtered):,}"
        )
    
    with col4:
        if "claim_amount" in filtered.columns:
            total_amount = pd.to_numeric(filtered["claim_amount"], errors="coerce").sum()
            st.metric(
                label="Total Amount at Risk",
                value=f"${total_amount:,.2f}"
            )
        else:
            st.metric(label="Total Amount at Risk", value="N/A")
    
    # Charts
    st.header("📊 Anomaly Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Anomalies by Type")
        filtered_counts = get_anomaly_counts(filtered)
        if len(filtered_counts) > 0:
            chart_data = pd.DataFrame({
                "Anomaly Type": filtered_counts.index,
                "Count": filtered_counts.values
            })
            st.bar_chart(chart_data.set_index("Anomaly Type"))
        else:
            st.info("No anomaly data to display")
    
    with col2:
        st.subheader("Anomalies by Provider (Top 10)")
        if "provider_id" in filtered.columns:
            provider_counts = filtered["provider_id"].value_counts().head(10)
            if len(provider_counts) > 0:
                chart_data = pd.DataFrame({
                    "Provider": provider_counts.index,
                    "Count": provider_counts.values
                })
                st.bar_chart(chart_data.set_index("Provider"))
            else:
                st.info("No provider data to display")
        else:
            st.info("Provider ID column not available")
    
    # Anomalies by status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Anomalies by Claim Status")
        if "claim_status" in filtered.columns:
            status_counts = filtered["claim_status"].value_counts()
            if len(status_counts) > 0:
                chart_data = pd.DataFrame({
                    "Status": status_counts.index,
                    "Count": status_counts.values
                })
                st.bar_chart(chart_data.set_index("Status"))
    
    with col2:
        st.subheader("Anomalies Over Time")
        if "service_date" in filtered.columns:
            time_data = filtered.copy()
            time_data["month"] = time_data["service_date"].dt.to_period("M").astype(str)
            monthly_counts = time_data.groupby("month").size()
            if len(monthly_counts) > 0:
                st.line_chart(monthly_counts)
    
    # Anomaly Table
    st.header("📋 Anomaly Details")
    
    # Select columns to display
    display_cols = ["claim_id", "member_id", "provider_id", "claim_amount", 
                    "service_date", "claim_status", "anomaly_reasons_str", "explanation"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    
    # Number of rows to show
    n_rows = st.slider("Number of rows to display", 10, 100, 20)
    
    # Sort options
    sort_col = st.selectbox("Sort by", ["num_flags", "claim_amount", "service_date", "claim_id"])
    sort_asc = st.checkbox("Ascending order", value=False)
    
    if sort_col in filtered.columns:
        display_df = filtered.sort_values(sort_col, ascending=sort_asc).head(n_rows)
    else:
        display_df = filtered.head(n_rows)
    
    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        height=400
    )
    
    # Explanation details
    st.header("💡 Anomaly Explanations")
    
    explained = filtered[filtered["explanation"].notna() & (filtered["explanation"] != "")]
    
    if len(explained) > 0:
        st.write(f"Showing explanations for {min(5, len(explained))} anomalies:")
        
        for idx, row in explained.head(5).iterrows():
            with st.expander(f"🔍 {row['claim_id']} - ${row.get('claim_amount', 'N/A')}"):
                st.write(f"**Provider:** {row.get('provider_id', 'N/A')}")
                st.write(f"**Service Date:** {row.get('service_date', 'N/A')}")
                st.write(f"**Status:** {row.get('claim_status', 'N/A')}")
                st.write(f"**Flags:** {row.get('anomaly_reasons_str', 'N/A')}")
                st.write("---")
                st.write(f"**Explanation:** {row.get('explanation', 'No explanation available')}")
    else:
        st.info("No explanations available. Run the pipeline with --explain-top option to generate explanations.")
    
    # Download section
    st.header("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Anomalies (CSV)",
            data=csv_data,
            file_name="filtered_anomalies.csv",
            mime="text/csv"
        )
    
    with col2:
        full_csv = anomalies.to_csv(index=False)
        st.download_button(
            label="Download All Anomalies (CSV)",
            data=full_csv,
            file_name="all_anomalies.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    app()