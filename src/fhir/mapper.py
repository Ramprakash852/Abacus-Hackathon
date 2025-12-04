"""
FHIR Compatibility Layer (basic)

Goal:
- Convert cleaned UH-DQIP tables (claims, providers, members) to simple FHIR resources:
  - Claim (fhir.Claim)
  - Patient (fhir.Patient)
  - Practitioner (fhir.Practitioner)
  - Encounter (fhir.Encounter) (simple)
- Provide mapping functions that take pandas Series / dict and return Python dicts matching FHIR JSON structure.
- Provide an `export_fhir_resources(claims_df, providers_df, members_df, out_dir="data/fhir")`
  which writes one JSON per resource per claim (or grouped files).
- Keep mappings minimalist but valid: include identifiers, references, status, total amount, service date, diagnosis/codes where present.
- Use standard system URIs where possible (e.g., "http://hl7.org/fhir/sid/icd-10" for ICD) — use config if provided.
- Include docstrings, type hints, and an example usage guarded by `if __name__ == "__main__":` that reads parquet from data/silver and writes JSON to data/fhir.
"""
from typing import Dict, Optional, Any
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Standard FHIR system URIs
DEFAULT_SYSTEMS = {
    "icd10": "http://hl7.org/fhir/sid/icd-10",
    "cpt": "http://www.ama-assn.org/go/cpt",
    "npi": "http://hl7.org/fhir/sid/us-npi",
    "member_id": "urn:oid:2.16.840.1.113883.3.8901.1",
    "claim_id": "urn:oid:2.16.840.1.113883.3.8901.2",
    "provider_id": "urn:oid:2.16.840.1.113883.3.8901.3",
}


def _safe_str(value: Any) -> Optional[str]:
    """Convert value to string, return None for NaN/None."""
    if pd.isna(value):
        return None
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float, return None for NaN/None."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def map_patient_to_fhir(member_row: pd.Series, id_system: str = "urn:uuid", mapping_config: dict = None) -> Dict:
    """
    Map a member row to a minimal FHIR Patient resource.
    
    Args:
        member_row: Pandas Series with member data
        id_system: System URI for identifier
        mapping_config: Optional custom mapping configuration
    
    Returns:
        Dictionary representing FHIR Patient resource
    """
    config = mapping_config or DEFAULT_SYSTEMS
    member_id = _safe_str(member_row.get("member_id"))
    
    patient = {
        "resourceType": "Patient",
        "id": member_id,
        "identifier": [
            {
                "system": config.get("member_id", id_system),
                "value": member_id
            }
        ],
        "active": True
    }
    
    # Add name if available
    first_name = _safe_str(member_row.get("first_name"))
    last_name = _safe_str(member_row.get("last_name"))
    if first_name or last_name:
        patient["name"] = [
            {
                "use": "official",
                "family": last_name,
                "given": [first_name] if first_name else []
            }
        ]
    
    # Add gender
    gender = _safe_str(member_row.get("gender"))
    if gender:
        gender_map = {"M": "male", "F": "female", "O": "other", "U": "unknown"}
        patient["gender"] = gender_map.get(gender.upper(), "unknown")
    
    # Add birth date
    dob = _safe_str(member_row.get("dob"))
    if dob:
        patient["birthDate"] = dob[:10] if len(dob) >= 10 else dob
    
    return patient


def map_practitioner_to_fhir(provider_row: pd.Series, id_system: str = "urn:uuid", mapping_config: dict = None) -> Dict:
    """
    Map a provider row to a minimal FHIR Practitioner resource.
    
    Args:
        provider_row: Pandas Series with provider data
        id_system: System URI for identifier
        mapping_config: Optional custom mapping configuration
    
    Returns:
        Dictionary representing FHIR Practitioner resource
    """
    config = mapping_config or DEFAULT_SYSTEMS
    provider_id = _safe_str(provider_row.get("provider_id"))
    
    practitioner = {
        "resourceType": "Practitioner",
        "id": provider_id,
        "identifier": [
            {
                "system": config.get("provider_id", id_system),
                "value": provider_id
            }
        ],
        "active": True
    }
    
    # Add NPI if available
    npi = _safe_str(provider_row.get("npi"))
    if npi:
        practitioner["identifier"].append({
            "system": config.get("npi", DEFAULT_SYSTEMS["npi"]),
            "value": npi
        })
    
    # Add name
    provider_name = _safe_str(provider_row.get("provider_name"))
    if provider_name:
        practitioner["name"] = [
            {
                "use": "official",
                "text": provider_name
            }
        ]
    
    # Add qualification/specialty
    specialty = _safe_str(provider_row.get("specialty"))
    if specialty:
        practitioner["qualification"] = [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "display": specialty
                        }
                    ],
                    "text": specialty
                }
            }
        ]
    
    # Add address (state)
    state = _safe_str(provider_row.get("state"))
    if state:
        practitioner["address"] = [
            {
                "use": "work",
                "state": state
            }
        ]
    
    return practitioner


def map_encounter_to_fhir(claim_row: pd.Series, member_row: pd.Series = None, mapping_config: dict = None) -> Dict:
    """
    Map a claim to a minimal FHIR Encounter resource.
    
    Args:
        claim_row: Pandas Series with claim data
        member_row: Optional Pandas Series with member data
        mapping_config: Optional custom mapping configuration
    
    Returns:
        Dictionary representing FHIR Encounter resource
    """
    claim_id = _safe_str(claim_row.get("claim_id"))
    member_id = _safe_str(claim_row.get("member_id"))
    provider_id = _safe_str(claim_row.get("provider_id"))
    service_date = _safe_str(claim_row.get("service_date"))
    
    # Map claim status to encounter status
    claim_status = _safe_str(claim_row.get("claim_status"))
    status_map = {
        "PAID": "finished",
        "DENIED": "cancelled",
        "PENDING": "in-progress",
        "APPEALED": "in-progress"
    }
    encounter_status = status_map.get(claim_status, "unknown") if claim_status else "unknown"
    
    encounter = {
        "resourceType": "Encounter",
        "id": f"enc-{claim_id}",
        "identifier": [
            {
                "system": "urn:encounter:id",
                "value": f"enc-{claim_id}"
            }
        ],
        "status": encounter_status,
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "AMB",
            "display": "ambulatory"
        }
    }
    
    # Add subject (patient reference)
    if member_id:
        encounter["subject"] = {
            "reference": f"Patient/{member_id}"
        }
    
    # Add participant (practitioner reference)
    if provider_id:
        encounter["participant"] = [
            {
                "individual": {
                    "reference": f"Practitioner/{provider_id}"
                }
            }
        ]
    
    # Add period
    if service_date:
        encounter["period"] = {
            "start": service_date,
            "end": service_date
        }
    
    return encounter


def map_claim_to_fhir(claim_row: pd.Series, member_row: pd.Series = None, provider_row: pd.Series = None, mapping_config: dict = None) -> Dict:
    """
    Map a claim row to a minimal FHIR Claim resource.
    
    Args:
        claim_row: Pandas Series with claim data
        member_row: Optional Pandas Series with member data
        provider_row: Optional Pandas Series with provider data
        mapping_config: Optional custom mapping configuration
    
    Returns:
        Dictionary representing FHIR Claim resource
    """
    config = mapping_config or DEFAULT_SYSTEMS
    
    claim_id = _safe_str(claim_row.get("claim_id"))
    member_id = _safe_str(claim_row.get("member_id"))
    provider_id = _safe_str(claim_row.get("provider_id"))
    claim_amount = _safe_float(claim_row.get("claim_amount"))
    service_date = _safe_str(claim_row.get("service_date"))
    icd_code = _safe_str(claim_row.get("icd_code"))
    cpt_code = _safe_str(claim_row.get("cpt_code"))
    claim_status = _safe_str(claim_row.get("claim_status"))
    
    # Map claim status to FHIR status
    status_map = {
        "PAID": "active",
        "DENIED": "cancelled",
        "PENDING": "draft",
        "APPEALED": "draft"
    }
    fhir_status = status_map.get(claim_status, "draft") if claim_status else "draft"
    
    claim = {
        "resourceType": "Claim",
        "id": claim_id,
        "identifier": [
            {
                "system": config.get("claim_id", DEFAULT_SYSTEMS["claim_id"]),
                "value": claim_id
            }
        ],
        "status": fhir_status,
        "type": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                    "code": "professional",
                    "display": "Professional"
                }
            ]
        },
        "use": "claim",
        "created": service_date or datetime.now().strftime("%Y-%m-%d")
    }
    
    # Add patient reference
    if member_id:
        claim["patient"] = {
            "reference": f"Patient/{member_id}"
        }
    
    # Add provider reference
    if provider_id:
        claim["provider"] = {
            "reference": f"Practitioner/{provider_id}"
        }
    
    # Add diagnosis (ICD code)
    if icd_code:
        claim["diagnosis"] = [
            {
                "sequence": 1,
                "diagnosisCodeableConcept": {
                    "coding": [
                        {
                            "system": config.get("icd10", DEFAULT_SYSTEMS["icd10"]),
                            "code": icd_code
                        }
                    ]
                }
            }
        ]
    
    # Add procedure (CPT code)
    if cpt_code:
        claim["procedure"] = [
            {
                "sequence": 1,
                "procedureCodeableConcept": {
                    "coding": [
                        {
                            "system": config.get("cpt", DEFAULT_SYSTEMS["cpt"]),
                            "code": cpt_code
                        }
                    ]
                }
            }
        ]
    
    # Add total
    if claim_amount is not None:
        claim["total"] = {
            "value": claim_amount,
            "currency": "USD"
        }
    
    # Add billable period
    if service_date:
        claim["billablePeriod"] = {
            "start": service_date,
            "end": service_date
        }
    
    # Add item (line item)
    claim["item"] = [
        {
            "sequence": 1,
            "productOrService": {
                "coding": [
                    {
                        "system": config.get("cpt", DEFAULT_SYSTEMS["cpt"]),
                        "code": cpt_code or "99999"
                    }
                ]
            },
            "servicedDate": service_date,
            "unitPrice": {
                "value": claim_amount or 0,
                "currency": "USD"
            },
            "net": {
                "value": claim_amount or 0,
                "currency": "USD"
            }
        }
    ]
    
    return claim


def export_fhir_resources(
    claims_df: pd.DataFrame, 
    providers_df: pd.DataFrame, 
    members_df: pd.DataFrame, 
    out_dir: str = "data/fhir",
    mapping_config: dict = None,
    max_claims: int = None
) -> Dict[str, int]:
    """
    Export claims, providers, and members to FHIR JSON resources.
    
    Args:
        claims_df: DataFrame with claims data
        providers_df: DataFrame with providers data
        members_df: DataFrame with members data
        out_dir: Output directory for FHIR resources
        mapping_config: Optional custom mapping configuration
        max_claims: Maximum number of claims to export (None for all)
    
    Returns:
        Dictionary with counts of exported resources
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (out_path / "Patient").mkdir(exist_ok=True)
    (out_path / "Practitioner").mkdir(exist_ok=True)
    (out_path / "Claim").mkdir(exist_ok=True)
    (out_path / "Encounter").mkdir(exist_ok=True)
    
    # Index providers and members for lookup
    providers_dict = {}
    if providers_df is not None and len(providers_df) > 0:
        providers_dict = {row["provider_id"]: row for _, row in providers_df.iterrows()}
    
    members_dict = {}
    if members_df is not None and len(members_df) > 0:
        members_dict = {row["member_id"]: row for _, row in members_df.iterrows()}
    
    # Track exported resources
    exported_patients = set()
    exported_practitioners = set()
    counts = {"Patient": 0, "Practitioner": 0, "Claim": 0, "Encounter": 0}
    
    # Limit claims if specified
    claims_to_export = claims_df.head(max_claims) if max_claims else claims_df
    
    print(f"Exporting FHIR resources to {out_path}...")
    
    for idx, claim_row in claims_to_export.iterrows():
        claim_id = _safe_str(claim_row.get("claim_id"))
        member_id = _safe_str(claim_row.get("member_id"))
        provider_id = _safe_str(claim_row.get("provider_id"))
        
        if not claim_id:
            continue
        
        # Export Patient (if not already exported)
        if member_id and member_id not in exported_patients:
            member_row = members_dict.get(member_id)
            if member_row is not None:
                patient = map_patient_to_fhir(member_row, mapping_config=mapping_config)
                with open(out_path / "Patient" / f"{member_id}.json", "w") as f:
                    json.dump(patient, f, indent=2)
                exported_patients.add(member_id)
                counts["Patient"] += 1
        
        # Export Practitioner (if not already exported)
        if provider_id and provider_id not in exported_practitioners:
            provider_row = providers_dict.get(provider_id)
            if provider_row is not None:
                practitioner = map_practitioner_to_fhir(provider_row, mapping_config=mapping_config)
                with open(out_path / "Practitioner" / f"{provider_id}.json", "w") as f:
                    json.dump(practitioner, f, indent=2)
                exported_practitioners.add(provider_id)
                counts["Practitioner"] += 1
        
        # Export Claim
        member_row = members_dict.get(member_id) if member_id else None
        provider_row = providers_dict.get(provider_id) if provider_id else None
        
        claim = map_claim_to_fhir(claim_row, member_row, provider_row, mapping_config)
        with open(out_path / "Claim" / f"{claim_id}.json", "w") as f:
            json.dump(claim, f, indent=2)
        counts["Claim"] += 1
        
        # Export Encounter
        encounter = map_encounter_to_fhir(claim_row, member_row, mapping_config)
        with open(out_path / "Encounter" / f"enc-{claim_id}.json", "w") as f:
            json.dump(encounter, f, indent=2)
        counts["Encounter"] += 1
    
    print(f"\nExport complete:")
    for resource_type, count in counts.items():
        print(f"  {resource_type}: {count} resources")
    
    return counts


if __name__ == "__main__":
    # Example usage: read from silver and export to FHIR
    silver_dir = Path("data/silver")
    
    print("Loading data from silver layer...")
    
    # Load data
    claims_df = pd.read_parquet(silver_dir / "claims.parquet") if (silver_dir / "claims.parquet").exists() else pd.DataFrame()
    providers_df = pd.read_parquet(silver_dir / "providers.parquet") if (silver_dir / "providers.parquet").exists() else pd.DataFrame()
    members_df = pd.read_parquet(silver_dir / "members.parquet") if (silver_dir / "members.parquet").exists() else pd.DataFrame()
    
    print(f"Loaded {len(claims_df)} claims, {len(providers_df)} providers, {len(members_df)} members")
    
    # Export to FHIR (limit to first 100 claims for demo)
    counts = export_fhir_resources(
        claims_df, 
        providers_df, 
        members_df,
        out_dir="data/fhir",
        max_claims=100
    )
    
    # Show sample output
    print("\nSample FHIR Claim resource:")
    fhir_dir = Path("data/fhir/Claim")
    sample_files = list(fhir_dir.glob("*.json"))[:1]
    if sample_files:
        with open(sample_files[0]) as f:
            sample = json.load(f)
        print(json.dumps(sample, indent=2)[:1000] + "...")