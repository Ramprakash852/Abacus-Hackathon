"""
DATA QUALITY RULES ENGINE (MVP)

- Implement rule functions:
  missing_mandatory_fields(df) -> boolean Series
  duplicate_claim_id(df) -> boolean Series
  invalid_dates(df) -> boolean Series
  negative_or_zero_amount(df) -> boolean Series
  invalid_icd_format(df) -> boolean Series
- Implement run_dq(silver_dir='data/silver', output_dir='data/gold') that:
  * loads cleaned claims
  * computes flags per row and produces flags_list and num_flags
  * writes anomalies to output_dir/anomalies.parquet and anomalies.csv
  * returns anomalies DataFrame
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re


def missing_mandatory_fields(df: pd.DataFrame) -> pd.Series:
    """Check for missing values in mandatory fields."""
    mandatory_fields = ["claim_id", "member_id", "provider_id", "claim_amount", "service_date"]
    missing_mask = df[mandatory_fields].isnull().any(axis=1)
    return missing_mask


def duplicate_claim_id(df: pd.DataFrame) -> pd.Series:
    """Identify duplicate claim IDs."""
    return df.duplicated(subset=["claim_id"], keep=False)


def invalid_dates(df: pd.DataFrame) -> pd.Series:
    """Check for invalid or missing service dates."""
    # Dates should be valid and not in the future
    dates = pd.to_datetime(df["service_date"], errors="coerce")
    today = pd.Timestamp.now()
    
    invalid_mask = dates.isnull() | (dates > today)
    return invalid_mask


def negative_or_zero_amount(df: pd.DataFrame) -> pd.Series:
    """Check for negative or zero claim amounts."""
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
    return (amounts <= 0) | amounts.isnull()


def invalid_icd_format(df: pd.DataFrame) -> pd.Series:
    """Check for invalid ICD-10 code format."""
    def is_invalid_icd(code):
        if pd.isna(code) or str(code) in ["nan", "None", ""]:
            return True
        code = str(code).strip().upper()
        # ICD-10 format: Letter followed by 2+ digits, optional decimal
        pattern = r"^[A-Z]\d{2}\.?\d*[A-Z]?$"
        return not bool(re.match(pattern, code))
    
    return df["icd_code"].apply(is_invalid_icd)


def invalid_cpt_format(df: pd.DataFrame) -> pd.Series:
    """Check for invalid CPT code format."""
    def is_invalid_cpt(code):
        if pd.isna(code) or str(code) in ["nan", "None", ""]:
            return True
        code = str(code).strip()
        # CPT codes are 5 digits
        return not bool(re.match(r"^\d{5}$", code))
    
    return df["cpt_code"].apply(is_invalid_cpt)


def outlier_amount(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Detect outliers in claim amounts using z-score method."""
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
    mean_amt = amounts.mean()
    std_amt = amounts.std()
    
    if std_amt == 0 or pd.isna(std_amt):
        return pd.Series([False] * len(df), index=df.index)
    
    z_scores = np.abs((amounts - mean_amt) / std_amt)
    return z_scores > threshold


# Define all DQ rules
DQ_RULES = {
    "missing_mandatory": missing_mandatory_fields,
    "duplicate_claim": duplicate_claim_id,
    "invalid_date": invalid_dates,
    "invalid_amount": negative_or_zero_amount,
    "invalid_icd": invalid_icd_format,
    "invalid_cpt": invalid_cpt_format,
    "outlier_amount": outlier_amount,
}


def run_dq(silver_dir="data/silver", output_dir="data/gold"):
    """
    Run data quality rules on silver layer claims and output anomalies.
    
    Args:
        silver_dir: Directory containing silver parquet files
        output_dir: Directory to write anomaly outputs (gold layer)
    
    Returns:
        DataFrame containing all flagged anomalies
    """
    silver_path = Path(silver_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load claims data
    claims_file = silver_path / "claims.parquet"
    if not claims_file.exists():
        print(f"Error: {claims_file} not found")
        return None
    
    print(f"Loading claims from {claims_file}...")
    df = pd.read_parquet(claims_file)
    print(f"  Loaded {len(df)} rows")
    
    # Apply all DQ rules
    print("\nApplying DQ rules...")
    flags_df = pd.DataFrame(index=df.index)
    
    for rule_name, rule_func in DQ_RULES.items():
        flags_df[rule_name] = rule_func(df)
        flagged_count = flags_df[rule_name].sum()
        print(f"  {rule_name}: {flagged_count} flagged ({flagged_count/len(df)*100:.1f}%)")
    
    # Create flags_list and num_flags columns
    def get_flags_list(row):
        flags = [col for col in flags_df.columns if row[col]]
        return ",".join(flags) if flags else ""
    
    df["flags_list"] = flags_df.apply(get_flags_list, axis=1)
    df["num_flags"] = flags_df.sum(axis=1)
    
    # Filter to only anomalies (rows with at least one flag)
    anomalies = df[df["num_flags"] > 0].copy()
    
    print(f"\nTotal anomalies: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}% of all claims)")
    
    # Write outputs
    parquet_file = output_path / "anomalies.parquet"
    csv_file = output_path / "anomalies.csv"
    
    anomalies.to_parquet(parquet_file, index=False)
    anomalies.to_csv(csv_file, index=False)
    
    print(f"\nOutputs written to:")
    print(f"  {parquet_file}")
    print(f"  {csv_file}")
    
    # Also write clean claims (no flags) to gold
    clean_claims = df[df["num_flags"] == 0].drop(columns=["flags_list", "num_flags"])
    clean_file = output_path / "claims_clean.parquet"
    clean_claims.to_parquet(clean_file, index=False)
    print(f"  {clean_file} ({len(clean_claims)} clean claims)")
    
    # Summary statistics
    print("\n=== DQ Summary ===")
    print(f"Total claims processed: {len(df)}")
    print(f"Clean claims: {len(clean_claims)} ({len(clean_claims)/len(df)*100:.1f}%)")
    print(f"Anomalous claims: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)")
    
    return anomalies


if __name__ == "__main__":
    run_dq()