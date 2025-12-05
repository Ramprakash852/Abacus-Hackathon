import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dq.rules import (
    missing_mandatory_fields,
    duplicate_claim_id,
    invalid_dates,
    negative_or_zero_amount,
    invalid_icd_format,
    invalid_cpt_format,
    outlier_amount,
    run_dq
)
from src.anomaly.detect import detect_anomalies


# ============================================================
# Test Data Fixtures
# ============================================================

@pytest.fixture
def sample_claims():
    """Create a sample claims DataFrame for testing."""
    return pd.DataFrame({
        "claim_id": ["CLM001", "CLM002", "CLM003", "CLM004", "CLM005"],
        "member_id": ["MBR001", "MBR002", "MBR003", "MBR004", "MBR005"],
        "provider_id": ["PRV001", "PRV002", "PRV003", "PRV004", "PRV005"],
        "claim_amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        "service_date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-01"],
        "icd_code": ["A00.0", "B20", "C34.90", "D50.9", "E11.9"],
        "cpt_code": ["99213", "99214", "99215", "99203", "99204"],
        "claim_status": ["PAID", "DENIED", "PENDING", "PAID", "PAID"]
    })


@pytest.fixture
def claims_with_missing_fields():
    """Claims with missing mandatory fields."""
    return pd.DataFrame({
        "claim_id": ["CLM001", "CLM002", "CLM003"],
        "member_id": ["MBR001", None, "MBR003"],
        "provider_id": ["PRV001", "PRV002", None],
        "claim_amount": [None, 200.0, 300.0],
        "service_date": ["2024-01-15", None, "2024-03-10"],
        "icd_code": ["A00.0", "B20", "C34.90"],
        "cpt_code": ["99213", "99214", "99215"],
        "claim_status": ["PAID", "DENIED", "PENDING"]
    })


@pytest.fixture
def claims_with_duplicates():
    """Claims with duplicate claim IDs."""
    return pd.DataFrame({
        "claim_id": ["CLM001", "CLM001", "CLM002", "CLM003", "CLM003"],
        "member_id": ["MBR001", "MBR001", "MBR002", "MBR003", "MBR003"],
        "provider_id": ["PRV001", "PRV001", "PRV002", "PRV003", "PRV003"],
        "claim_amount": [100.0, 100.0, 200.0, 300.0, 300.0],
        "service_date": ["2024-01-15", "2024-01-15", "2024-02-20", "2024-03-10", "2024-03-10"],
        "icd_code": ["A00.0", "A00.0", "B20", "C34.90", "C34.90"],
        "cpt_code": ["99213", "99213", "99214", "99215", "99215"],
        "claim_status": ["PAID", "PAID", "DENIED", "PENDING", "PENDING"]
    })


@pytest.fixture
def claims_with_outliers():
    """Claims with outlier amounts - need more data points for z-score to work."""
    return pd.DataFrame({
        "claim_id": [f"CLM{str(i).zfill(3)}" for i in range(1, 21)] + ["CLM021"],
        "member_id": [f"MBR{str(i).zfill(3)}" for i in range(1, 22)],
        "provider_id": [f"PRV{str((i % 5) + 1).zfill(3)}" for i in range(1, 22)],
        "claim_amount": [100.0, 150.0, 120.0, 180.0, 130.0, 
                         110.0, 160.0, 140.0, 170.0, 125.0,
                         115.0, 155.0, 145.0, 135.0, 165.0,
                         105.0, 175.0, 185.0, 195.0, 190.0,
                         500000.0],  # CLM021 is a massive outlier
        "service_date": ["2024-01-15"] * 21,
        "icd_code": ["A00.0"] * 21,
        "cpt_code": ["99213"] * 21,
        "claim_status": ["PAID"] * 21
    })


# ============================================================
# Test Missing Mandatory Fields
# ============================================================

def test_missing_fields_flag(claims_with_missing_fields):
    """Test that missing mandatory fields are flagged."""
    result = missing_mandatory_fields(claims_with_missing_fields)
    
    # All 3 rows have at least one missing mandatory field
    assert result.sum() == 3
    assert result.iloc[0] == True  # missing claim_amount
    assert result.iloc[1] == True  # missing member_id and service_date
    assert result.iloc[2] == True  # missing provider_id


def test_no_missing_fields(sample_claims):
    """Test that complete claims are not flagged for missing fields."""
    result = missing_mandatory_fields(sample_claims)
    assert result.sum() == 0


# ============================================================
# Test Duplicate Claims
# ============================================================

def test_duplicate_claim_flag(claims_with_duplicates):
    """Test that duplicate claim IDs are flagged."""
    result = duplicate_claim_id(claims_with_duplicates)
    
    # CLM001 appears twice, CLM003 appears twice = 4 flagged
    assert result.sum() == 4
    assert result.iloc[0] == True  # CLM001 (first)
    assert result.iloc[1] == True  # CLM001 (second)
    assert result.iloc[2] == False  # CLM002 (unique)
    assert result.iloc[3] == True  # CLM003 (first)
    assert result.iloc[4] == True  # CLM003 (second)


def test_no_duplicates(sample_claims):
    """Test that unique claims are not flagged as duplicates."""
    result = duplicate_claim_id(sample_claims)
    assert result.sum() == 0


# ============================================================
# Test Invalid Dates
# ============================================================

def test_invalid_dates():
    """Test that invalid dates are flagged."""
    df = pd.DataFrame({
        "claim_id": ["CLM001", "CLM002", "CLM003", "CLM004"],
        "member_id": ["MBR001", "MBR002", "MBR003", "MBR004"],
        "provider_id": ["PRV001", "PRV002", "PRV003", "PRV004"],
        "claim_amount": [100.0, 200.0, 300.0, 400.0],
        "service_date": ["2024-01-15", "invalid-date", None, "2099-12-31"],  # Future date
        "icd_code": ["A00.0", "B20", "C34.90", "D50.9"],
        "cpt_code": ["99213", "99214", "99215", "99203"],
        "claim_status": ["PAID", "DENIED", "PENDING", "PAID"]
    })
    
    result = invalid_dates(df)
    
    assert result.iloc[0] == False  # Valid date
    assert result.iloc[1] == True   # Invalid format
    assert result.iloc[2] == True   # Missing
    assert result.iloc[3] == True   # Future date


# ============================================================
# Test Invalid Amounts
# ============================================================

def test_negative_or_zero_amount():
    """Test that negative and zero amounts are flagged."""
    df = pd.DataFrame({
        "claim_id": ["CLM001", "CLM002", "CLM003", "CLM004", "CLM005"],
        "member_id": ["MBR001", "MBR002", "MBR003", "MBR004", "MBR005"],
        "provider_id": ["PRV001", "PRV002", "PRV003", "PRV004", "PRV005"],
        "claim_amount": [100.0, 0.0, -50.0, None, 200.0],
        "service_date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-01"],
        "icd_code": ["A00.0", "B20", "C34.90", "D50.9", "E11.9"],
        "cpt_code": ["99213", "99214", "99215", "99203", "99204"],
        "claim_status": ["PAID", "DENIED", "PENDING", "PAID", "PAID"]
    })
    
    result = negative_or_zero_amount(df)
    
    assert result.iloc[0] == False  # Valid positive amount
    assert result.iloc[1] == True   # Zero
    assert result.iloc[2] == True   # Negative
    assert result.iloc[3] == True   # Missing
    assert result.iloc[4] == False  # Valid positive amount


# ============================================================
# Test Invalid ICD Codes
# ============================================================

def test_invalid_icd_format():
    """Test that invalid ICD codes are flagged."""
    df = pd.DataFrame({
        "claim_id": ["CLM001", "CLM002", "CLM003", "CLM004", "CLM005"],
        "member_id": ["MBR001", "MBR002", "MBR003", "MBR004", "MBR005"],
        "provider_id": ["PRV001", "PRV002", "PRV003", "PRV004", "PRV005"],
        "claim_amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        "service_date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-01"],
        "icd_code": ["A00.0", "INVALID", "123", None, "E11.9"],
        "cpt_code": ["99213", "99214", "99215", "99203", "99204"],
        "claim_status": ["PAID", "DENIED", "PENDING", "PAID", "PAID"]
    })
    
    result = invalid_icd_format(df)
    
    assert result.iloc[0] == False  # Valid ICD
    assert result.iloc[1] == True   # Invalid format
    assert result.iloc[2] == True   # Numeric only
    assert result.iloc[3] == True   # Missing
    assert result.iloc[4] == False  # Valid ICD


# ============================================================
# Test Invalid CPT Codes
# ============================================================

def test_invalid_cpt_format():
    """Test that invalid CPT codes are flagged."""
    df = pd.DataFrame({
        "claim_id": ["CLM001", "CLM002", "CLM003", "CLM004", "CLM005"],
        "member_id": ["MBR001", "MBR002", "MBR003", "MBR004", "MBR005"],
        "provider_id": ["PRV001", "PRV002", "PRV003", "PRV004", "PRV005"],
        "claim_amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        "service_date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-01"],
        "icd_code": ["A00.0", "B20", "C34.90", "D50.9", "E11.9"],
        "cpt_code": ["99213", "XXXXX", "123", None, "99204"],
        "claim_status": ["PAID", "DENIED", "PENDING", "PAID", "PAID"]
    })
    
    result = invalid_cpt_format(df)
    
    assert result.iloc[0] == False  # Valid CPT (5 digits)
    assert result.iloc[1] == True   # Invalid format (letters)
    assert result.iloc[2] == True   # Invalid format (3 digits)
    assert result.iloc[3] == True   # Missing
    assert result.iloc[4] == False  # Valid CPT


# ============================================================
# Test Outlier Detection
# ============================================================

def test_amount_outlier(claims_with_outliers):
    """Test that known outliers are flagged by outlier detection."""
    result = outlier_amount(claims_with_outliers, threshold=3.0)
    
    # CLM021 with 500000 should be flagged as outlier (last row, index 20)
    assert result.iloc[20] == True  # CLM021 is outlier
    # Normal claims should not be flagged
    assert result.iloc[0] == False
    assert result.iloc[5] == False
    assert result.iloc[10] == False


def test_no_outliers(sample_claims):
    """Test that normal claims are not flagged as outliers."""
    result = outlier_amount(sample_claims, threshold=3.0)
    # No extreme outliers in sample data
    assert result.sum() <= 1  # At most 1 might be flagged depending on distribution


# ============================================================
# Test Anomaly Detection Integration
# ============================================================

def test_detect_anomalies_with_outliers(claims_with_outliers):
    """Test that detect_anomalies correctly identifies outliers."""
    result = detect_anomalies(claims_with_outliers, zscore_threshold=3.0)
    
    assert "is_anomalous" in result.columns
    assert "anomaly_reasons" in result.columns
    
    # The outlier claim (CLM021) should be flagged
    outlier_row = result[result["claim_id"] == "CLM021"]
    assert outlier_row["is_anomalous"].iloc[0] == True


def test_detect_anomalies_empty_df():
    """Test that detect_anomalies handles empty DataFrame."""
    empty_df = pd.DataFrame()
    result = detect_anomalies(empty_df)
    assert result is None or len(result) == 0


def test_detect_anomalies_adds_columns(sample_claims):
    """Test that detect_anomalies adds required columns."""
    result = detect_anomalies(sample_claims)
    
    assert "is_anomalous" in result.columns
    assert "anomaly_reasons" in result.columns
    assert "anomaly_reasons_str" in result.columns


# ============================================================
# Test Full DQ Pipeline
# ============================================================

def test_run_dq_creates_output(tmp_path):
    """Test that run_dq creates output files."""
    # Create test silver data
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    
    test_claims = pd.DataFrame({
        "claim_id": ["CLM001", "CLM001", "CLM003"],  # Duplicate
        "member_id": ["MBR001", "MBR002", None],      # Missing
        "provider_id": ["PRV001", "PRV002", "PRV003"],
        "claim_amount": [100.0, -50.0, 300.0],        # Negative
        "service_date": ["2024-01-15", "2024-02-20", "invalid"],
        "icd_code": ["A00.0", "INVALID", "C34.90"],
        "cpt_code": ["99213", "99214", "123"],
        "claim_status": ["PAID", "DENIED", "PENDING"]
    })
    test_claims.to_parquet(silver_dir / "claims.parquet")
    
    # Run DQ
    output_dir = tmp_path / "gold"
    result = run_dq(str(silver_dir), str(output_dir))
    
    # Check outputs
    assert result is not None
    assert len(result) > 0
    assert (output_dir / "anomalies.parquet").exists()
    assert (output_dir / "anomalies.csv").exists()


# ============================================================
# Edge Cases
# ============================================================

def test_all_valid_claims(sample_claims):
    """Test processing of completely valid claims."""
    # Check individual rules
    assert missing_mandatory_fields(sample_claims).sum() == 0
    assert duplicate_claim_id(sample_claims).sum() == 0
    assert negative_or_zero_amount(sample_claims).sum() == 0


def test_single_row_dataframe():
    """Test DQ rules with single-row DataFrame."""
    df = pd.DataFrame({
        "claim_id": ["CLM001"],
        "member_id": ["MBR001"],
        "provider_id": ["PRV001"],
        "claim_amount": [100.0],
        "service_date": ["2024-01-15"],
        "icd_code": ["A00.0"],
        "cpt_code": ["99213"],
        "claim_status": ["PAID"]
    })
    
    # All rules should work with single row
    assert len(missing_mandatory_fields(df)) == 1
    assert len(duplicate_claim_id(df)) == 1
    assert len(invalid_dates(df)) == 1
    assert len(negative_or_zero_amount(df)) == 1