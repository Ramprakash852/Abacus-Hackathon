"""
TRANSFORM MODULE (pandas)

- Function: run_transform(bronze_dir='data/bronze', silver_dir='data/silver')
- Read bronze parquet files (claims, providers)
- Normalize dates to ISO, trim/clean strings, cast amounts to float (coerce errors)
- Replace negative amounts with NaN
- Standardize ICD/CPT formats via regex (basic normalization)
- Write cleaned parquet files to silver_dir and return DataFrames
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from all string columns."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["nan", "None", ""], np.nan)
    return df


def normalize_date(date_series: pd.Series) -> pd.Series:
    """Normalize dates to ISO format (YYYY-MM-DD), invalid dates become NaT."""
    return pd.to_datetime(date_series, errors="coerce").dt.strftime("%Y-%m-%d")


def normalize_icd_code(code: str) -> str:
    """
    Normalize ICD-10 codes to standard format.
    Valid ICD-10 format: Letter followed by 2 digits, optionally followed by decimal and more digits.
    """
    if pd.isna(code) or code in ["nan", "None", ""]:
        return np.nan
    
    code = str(code).strip().upper()
    
    # Basic ICD-10 pattern: starts with letter, followed by digits, optional decimal
    icd_pattern = r"^[A-Z]\d{2}\.?\d*[A-Z]?$"
    if re.match(icd_pattern, code):
        return code
    
    return np.nan  # Invalid format


def normalize_cpt_code(code: str) -> str:
    """
    Normalize CPT codes to standard format.
    Valid CPT format: 5 digits.
    """
    if pd.isna(code) or code in ["nan", "None", ""]:
        return np.nan
    
    code = str(code).strip()
    
    # CPT codes are 5 digits
    if re.match(r"^\d{5}$", code):
        return code
    
    return np.nan  # Invalid format


def transform_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations to claims data."""
    df = df.copy()
    
    # Clean string columns
    df = clean_string_columns(df)
    
    # Normalize dates
    df["service_date"] = normalize_date(df["service_date"])
    
    # Cast claim_amount to float, coerce errors
    df["claim_amount"] = pd.to_numeric(df["claim_amount"], errors="coerce")
    
    # Replace negative amounts with NaN
    df.loc[df["claim_amount"] < 0, "claim_amount"] = np.nan
    
    # Normalize ICD and CPT codes
    df["icd_code"] = df["icd_code"].apply(normalize_icd_code)
    df["cpt_code"] = df["cpt_code"].apply(normalize_cpt_code)
    
    # Standardize claim_status to uppercase
    df["claim_status"] = df["claim_status"].str.upper()
    
    return df


def transform_providers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations to providers data."""
    df = df.copy()
    df = clean_string_columns(df)
    
    # Standardize state to uppercase
    df["state"] = df["state"].str.upper()
    
    return df


def transform_members(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations to members data."""
    df = df.copy()
    df = clean_string_columns(df)
    
    # Normalize DOB
    df["dob"] = normalize_date(df["dob"])
    
    # Standardize gender
    df["gender"] = df["gender"].str.upper()
    
    return df


def run_transform(bronze_dir="data/bronze", silver_dir="data/silver"):
    """
    Transform bronze layer data and write to silver layer.
    
    Args:
        bronze_dir: Directory containing bronze parquet files
        silver_dir: Directory to write transformed parquet files
    
    Returns:
        Dictionary mapping file type to transformed DataFrame
    """
    bronze_path = Path(bronze_dir)
    silver_path = Path(silver_dir)
    silver_path.mkdir(parents=True, exist_ok=True)
    
    transforms = {
        "claims": transform_claims,
        "providers": transform_providers,
        "members": transform_members
    }
    
    results = {}
    
    for file_type, transform_func in transforms.items():
        parquet_file = bronze_path / f"{file_type}.parquet"
        
        if not parquet_file.exists():
            print(f"Warning: {parquet_file} not found, skipping...")
            continue
        
        # Read bronze data
        print(f"Transforming {file_type}...")
        df = pd.read_parquet(parquet_file)
        original_count = len(df)
        
        # Apply transformations
        df_transformed = transform_func(df)
        
        # Report on null values introduced
        null_counts = df_transformed.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            print(f"  Null values after transform:")
            for col, count in cols_with_nulls.items():
                print(f"    {col}: {count} ({count/len(df_transformed)*100:.1f}%)")
        
        # Write to silver
        output_file = silver_path / f"{file_type}.parquet"
        df_transformed.to_parquet(output_file, index=False)
        print(f"  Written {len(df_transformed)} rows to {output_file}")
        
        results[file_type] = df_transformed
    
    print(f"\nTransform complete. Files written to {silver_path}")
    return results


if __name__ == "__main__":
    run_transform()