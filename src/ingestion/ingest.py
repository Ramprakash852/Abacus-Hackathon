import pandas as pd
from pathlib import Path

# Required columns for each file type
REQUIRED_COLUMNS = {
    "claims": ["claim_id", "member_id", "provider_id", "claim_amount", "service_date", "icd_code", "cpt_code", "claim_status"],
    "providers": ["provider_id", "provider_name", "specialty", "state", "npi"],
    "members": ["member_id", "first_name", "last_name", "dob", "gender", "plan_type"]
}

def validate_columns(df: pd.DataFrame, file_type: str) -> bool:
    """Validate that required columns are present in the DataFrame."""
    required = REQUIRED_COLUMNS.get(file_type, [])
    missing = set(required) - set(df.columns)
    if missing:
        print(f"Warning: {file_type} missing required columns: {missing}")
        return False
    return True

def run_ingest(input_dir="data", output_dir="data/bronze"):
    """
    Ingest CSV files and write to parquet format in bronze layer.
    
    Args:
        input_dir: Directory containing source CSV files
        output_dir: Directory to write parquet files (bronze layer)
    
    Returns:
        Dictionary mapping file type to output path
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files_to_ingest = ["claims", "providers", "members"]
    output_paths = {}
    
    for file_type in files_to_ingest:
        csv_file = input_path / f"{file_type}.csv"
        
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        # Read CSV
        print(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate columns
        validate_columns(df, file_type)
        
        # Try to write parquet, fallback to CSV
        try:
            import pyarrow
            output_file = output_path / f"{file_type}.parquet"
            df.to_parquet(output_file, index=False, engine="pyarrow")
            print(f"  Written to {output_file}")
        except ImportError:
            print("  pyarrow not available, falling back to CSV")
            output_file = output_path / f"{file_type}.csv"
            df.to_csv(output_file, index=False)
            print(f"  Written to {output_file}")
        
        output_paths[file_type] = str(output_file)
    
    print(f"\nIngestion complete. Files written to {output_path}")
    return output_paths


if __name__ == "__main__":
    run_ingest()