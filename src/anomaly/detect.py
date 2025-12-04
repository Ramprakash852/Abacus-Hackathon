"""
ANOMALY DETECTION (pandas)

- Function: detect_anomalies(df) where df is cleaned claims or DQ-annotated df
- Compute z-score of claim_amount grouped by provider_id or cpt_code
- Flag rows where z > 3 as amount_outlier
- Combine DQ flags into final is_anomalous boolean
- Add anomaly_reasons list per row
- Return DataFrame with is_anomalous and anomaly_reasons
"""
import pandas as pd
import numpy as np


def compute_zscore_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    """
    Compute z-scores for a value column grouped by another column.
    
    Args:
        df: DataFrame containing the data
        value_col: Column to compute z-scores for
        group_col: Column to group by
    
    Returns:
        Series of z-scores
    """
    values = pd.to_numeric(df[value_col], errors="coerce")
    
    # Compute group statistics
    group_stats = df.groupby(group_col)[value_col].agg(["mean", "std"])
    
    # Map back to original dataframe
    means = df[group_col].map(group_stats["mean"])
    stds = df[group_col].map(group_stats["std"])
    
    # Handle cases where std is 0 or NaN
    stds = stds.replace(0, np.nan)
    
    z_scores = (values - means) / stds
    return z_scores.fillna(0)


def detect_amount_outliers_by_provider(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Detect outliers in claim amount grouped by provider."""
    z_scores = compute_zscore_by_group(df, "claim_amount", "provider_id")
    return np.abs(z_scores) > threshold


def detect_amount_outliers_by_cpt(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Detect outliers in claim amount grouped by CPT code."""
    z_scores = compute_zscore_by_group(df, "claim_amount", "cpt_code")
    return np.abs(z_scores) > threshold


def detect_amount_outliers_global(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Detect global outliers in claim amount using z-score."""
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
    mean_amt = amounts.mean()
    std_amt = amounts.std()
    
    if std_amt == 0 or pd.isna(std_amt):
        return pd.Series([False] * len(df), index=df.index)
    
    z_scores = np.abs((amounts - mean_amt) / std_amt)
    return z_scores > threshold


def detect_anomalies(df: pd.DataFrame, zscore_threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies using z-score analysis and combine with existing DQ flags.
    
    Args:
        df: DataFrame with claims data (may include DQ flags from run_dq)
        zscore_threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with is_anomalous flag and anomaly_reasons column
    """
    if df is None or len(df) == 0:
        print("Warning: Empty or None DataFrame provided")
        return df
    
    df = df.copy()
    
    print(f"\nRunning anomaly detection on {len(df)} rows...")
    
    # Compute z-score based outlier flags
    anomaly_flags = {}
    
    # Global amount outliers
    anomaly_flags["zscore_global"] = detect_amount_outliers_global(df, zscore_threshold)
    print(f"  Z-score global outliers: {anomaly_flags['zscore_global'].sum()}")
    
    # Provider-level outliers (if provider_id exists and has valid values)
    if "provider_id" in df.columns and df["provider_id"].notna().sum() > 0:
        anomaly_flags["zscore_by_provider"] = detect_amount_outliers_by_provider(df, zscore_threshold)
        print(f"  Z-score by provider outliers: {anomaly_flags['zscore_by_provider'].sum()}")
    
    # CPT-level outliers (if cpt_code exists and has valid values)
    if "cpt_code" in df.columns and df["cpt_code"].notna().sum() > 0:
        anomaly_flags["zscore_by_cpt"] = detect_amount_outliers_by_cpt(df, zscore_threshold)
        print(f"  Z-score by CPT outliers: {anomaly_flags['zscore_by_cpt'].sum()}")
    
    # Build anomaly reasons
    def build_anomaly_reasons(row_idx):
        reasons = []
        
        # Include existing DQ flags if present
        if "flags_list" in df.columns:
            existing_flags = df.loc[row_idx, "flags_list"]
            if pd.notna(existing_flags) and existing_flags:
                reasons.extend(existing_flags.split(","))
        
        # Add z-score flags
        for flag_name, flag_series in anomaly_flags.items():
            if flag_series.loc[row_idx]:
                reasons.append(flag_name)
        
        return reasons
    
    # Apply anomaly detection
    df["anomaly_reasons"] = df.index.map(lambda idx: build_anomaly_reasons(idx))
    df["anomaly_reasons_str"] = df["anomaly_reasons"].apply(lambda x: ",".join(x) if x else "")
    
    # Determine is_anomalous
    df["is_anomalous"] = df["anomaly_reasons"].apply(lambda x: len(x) > 0)
    
    # Count anomalies
    total_anomalies = df["is_anomalous"].sum()
    print(f"\nTotal anomalies detected: {total_anomalies} ({total_anomalies/len(df)*100:.1f}%)")
    
    # Summary by anomaly type
    print("\nAnomaly breakdown:")
    all_reasons = []
    for reasons in df["anomaly_reasons"]:
        all_reasons.extend(reasons)
    
    if all_reasons:
        reason_counts = pd.Series(all_reasons).value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
    
    return df


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, ".")
    from src.dq.rules import run_dq
    
    anomalies = run_dq()
    if anomalies is not None:
        result = detect_anomalies(anomalies)
        print("\nSample output:")
        print(result[["claim_id", "claim_amount", "is_anomalous", "anomaly_reasons_str"]].head(10))