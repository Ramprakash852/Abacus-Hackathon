"""
GENERATE SYNTHETIC CLAIMS DATA (pandas)

Goal:
- Produce realistic CSVs for testing: data/claims.csv, data/providers.csv, data/members.csv
- claims schema: claim_id, member_id, provider_id, claim_amount, service_date, icd_code, cpt_code, claim_status
- Inject realistic errors: duplicates, missing fields, invalid dates/formats, outliers in claim_amount, invalid ICD/CPT formats
- Provide function `generate_synthetic_claims(num_claims=2000, out_dir="data")` that writes CSV files to out_dir
- When run as __main__, call generate_synthetic_claims()
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random
import string
from datetime import datetime, timedelta

def generate_synthetic_claims(num_claims=2000, out_dir="data"):
    """Generate synthetic claims, providers, and members data with intentional anomalies."""
    np.random.seed(42)
    random.seed(42)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate providers
    num_providers = 50
    providers = pd.DataFrame({
        "provider_id": [f"PRV{str(i).zfill(5)}" for i in range(1, num_providers + 1)],
        "provider_name": [f"Provider_{i}" for i in range(1, num_providers + 1)],
        "specialty": np.random.choice(["Cardiology", "Orthopedics", "General", "Neurology", "Oncology", "Pediatrics"], num_providers),
        "state": np.random.choice(["CA", "TX", "NY", "FL", "IL", "PA", "OH"], num_providers),
        "npi": [f"{random.randint(1000000000, 9999999999)}" for _ in range(num_providers)]
    })
    providers.to_csv(out_path / "providers.csv", index=False)
    
    # Generate members
    num_members = 500
    members = pd.DataFrame({
        "member_id": [f"MBR{str(i).zfill(6)}" for i in range(1, num_members + 1)],
        "first_name": [f"First_{i}" for i in range(1, num_members + 1)],
        "last_name": [f"Last_{i}" for i in range(1, num_members + 1)],
        "dob": pd.to_datetime(np.random.choice(pd.date_range("1950-01-01", "2005-12-31"), num_members)),
        "gender": np.random.choice(["M", "F"], num_members),
        "plan_type": np.random.choice(["HMO", "PPO", "EPO", "POS"], num_members)
    })
    members.to_csv(out_path / "members.csv", index=False)
    
    # Valid ICD-10 and CPT code patterns
    icd_codes = ["A00.0", "B20", "C34.90", "D50.9", "E11.9", "F32.9", "G43.909", "H26.9", "I10", "J06.9", 
                 "K21.0", "L50.9", "M54.5", "N39.0", "O80", "R10.9", "S72.001A", "T78.40XA", "Z00.00"]
    cpt_codes = ["99213", "99214", "99215", "99203", "99204", "99205", "36415", "80053", "85025", 
                 "71046", "93000", "90834", "97110", "99385", "99395"]
    
    # Generate base claims
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    claims = pd.DataFrame({
        "claim_id": [f"CLM{str(i).zfill(8)}" for i in range(1, num_claims + 1)],
        "member_id": np.random.choice(members["member_id"], num_claims),
        "provider_id": np.random.choice(providers["provider_id"], num_claims),
        "claim_amount": np.round(np.random.lognormal(mean=5, sigma=1.2, size=num_claims), 2),
        "service_date": [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_claims)],
        "icd_code": np.random.choice(icd_codes, num_claims),
        "cpt_code": np.random.choice(cpt_codes, num_claims),
        "claim_status": np.random.choice(["PAID", "DENIED", "PENDING", "APPEALED"], num_claims, p=[0.7, 0.15, 0.1, 0.05])
    })
    
    # === INJECT ANOMALIES ===
    
    # 1. Duplicate claims (~2%)
    num_duplicates = int(num_claims * 0.02)
    duplicate_indices = np.random.choice(claims.index, num_duplicates, replace=False)
    duplicates = claims.loc[duplicate_indices].copy()
    claims = pd.concat([claims, duplicates], ignore_index=True)
    
    # 2. Missing fields (~3% of rows)
    num_missing = int(len(claims) * 0.03)
    for col in ["member_id", "provider_id", "claim_amount", "icd_code"]:
        missing_indices = np.random.choice(claims.index, num_missing // 4, replace=False)
        claims.loc[missing_indices, col] = np.nan
    
    # 3. Invalid date formats (~1%)
    num_invalid_dates = int(len(claims) * 0.01)
    invalid_date_indices = np.random.choice(claims.index, num_invalid_dates, replace=False)
    invalid_dates = ["2024/13/45", "not-a-date", "31-02-2024", "2024-00-15", ""]
    claims["service_date"] = claims["service_date"].astype(str)
    claims.loc[invalid_date_indices, "service_date"] = np.random.choice(invalid_dates, num_invalid_dates)
    
    # 4. Outliers in claim_amount (~1%) - extremely high values
    num_outliers = int(len(claims) * 0.01)
    outlier_indices = np.random.choice(claims.index, num_outliers, replace=False)
    claims.loc[outlier_indices, "claim_amount"] = np.random.uniform(50000, 500000, num_outliers)
    
    # 5. Invalid ICD/CPT formats (~2%)
    num_invalid_codes = int(len(claims) * 0.02)
    invalid_icd_indices = np.random.choice(claims.index, num_invalid_codes, replace=False)
    invalid_cpt_indices = np.random.choice(claims.index, num_invalid_codes, replace=False)
    invalid_icds = ["INVALID", "123", "ZZZ.ZZ", "A", "12345678"]
    invalid_cpts = ["XXXXX", "123", "ABCDE", "0", "999999"]
    claims.loc[invalid_icd_indices, "icd_code"] = np.random.choice(invalid_icds, num_invalid_codes)
    claims.loc[invalid_cpt_indices, "cpt_code"] = np.random.choice(invalid_cpts, num_invalid_codes)
    
    # 6. Negative claim amounts (~0.5%)
    num_negative = int(len(claims) * 0.005)
    negative_indices = np.random.choice(claims.index, num_negative, replace=False)
    claims.loc[negative_indices, "claim_amount"] = -np.abs(claims.loc[negative_indices, "claim_amount"])
    
    # Shuffle and save
    claims = claims.sample(frac=1, random_state=42).reset_index(drop=True)
    claims.to_csv(out_path / "claims.csv", index=False)
    
    print(f"Generated {len(claims)} claims (including anomalies)")
    print(f"Generated {len(providers)} providers")
    print(f"Generated {len(members)} members")
    print(f"Files saved to: {out_path.absolute()}")


if __name__ == "__main__":
    generate_synthetic_claims()