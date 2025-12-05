import os
from typing import Optional


# Flag descriptions for fallback explanations
FLAG_DESCRIPTIONS = {
    "missing_mandatory": "Missing required fields (member_id, provider_id, claim_amount, or service_date)",
    "duplicate_claim": "This claim ID appears multiple times in the dataset",
    "invalid_date": "Service date is missing, invalid, or in the future",
    "invalid_amount": "Claim amount is missing, zero, or negative",
    "invalid_icd": "ICD-10 diagnosis code is missing or has invalid format",
    "invalid_cpt": "CPT procedure code is missing or has invalid format",
    "outlier_amount": "Claim amount is a statistical outlier (z-score > 3)",
    "zscore_global": "Claim amount is unusually high/low compared to all claims",
    "zscore_by_provider": "Claim amount is unusual for this provider's typical billing",
    "zscore_by_cpt": "Claim amount is unusual for this procedure code",
    "amount_outlier": "Claim amount is significantly outside normal range",
    "missing_provider": "Provider ID is missing from this claim",
}

# Remediation suggestions
FLAG_REMEDIATIONS = {
    "missing_mandatory": "Verify source data and ensure all required fields are populated before submission.",
    "duplicate_claim": "Check for duplicate submissions and verify if this is a valid resubmission or error.",
    "invalid_date": "Verify the service date with the provider and correct the date format.",
    "invalid_amount": "Review billing records and correct the claim amount to reflect actual charges.",
    "invalid_icd": "Validate the diagnosis code against ICD-10 reference and correct the format.",
    "invalid_cpt": "Verify the procedure code against CPT reference and ensure 5-digit format.",
    "outlier_amount": "Review for potential billing errors, upcoding, or legitimate high-cost services.",
    "zscore_global": "Investigate if the amount reflects actual services or requires adjustment.",
    "zscore_by_provider": "Compare with provider's historical billing patterns for this service type.",
    "zscore_by_cpt": "Verify the amount aligns with typical reimbursement for this procedure.",
    "amount_outlier": "Review claim details and verify amount against service documentation.",
    "missing_provider": "Obtain and add the provider NPI/ID before processing this claim.",
}


def build_prompt(record: dict) -> str:
    """Build a prompt for the LLM based on the anomaly record."""
    claim_id = record.get("claim_id", "Unknown")
    claim_amount = record.get("claim_amount", "N/A")
    provider_id = record.get("provider_id", "N/A")
    member_id = record.get("member_id", "N/A")
    service_date = record.get("service_date", "N/A")
    icd_code = record.get("icd_code", "N/A")
    cpt_code = record.get("cpt_code", "N/A")
    
    # Get flags from various possible field names
    flags = record.get("flags", [])
    if not flags:
        flags_str = record.get("flags_list", "") or record.get("anomaly_reasons_str", "")
        if flags_str:
            flags = flags_str.split(",")
    
    flags_text = ", ".join(flags) if flags else "Unknown"
    
    prompt = f"""You are a healthcare claims analyst. Analyze this flagged claim and provide a brief explanation.

Claim Details:
- Claim ID: {claim_id}
- Claim Amount: ${claim_amount}
- Provider ID: {provider_id}
- Member ID: {member_id}
- Service Date: {service_date}
- ICD-10 Code: {icd_code}
- CPT Code: {cpt_code}
- Flags: {flags_text}

Provide a 2-sentence explanation of why this claim was flagged, followed by 1 specific remediation suggestion.
Keep the response concise and actionable."""

    return prompt


def call_openai(prompt: str) -> Optional[str]:
    """Call OpenAI API to get explanation."""
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a healthcare claims analyst providing brief, actionable explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None


def generate_fallback_explanation(record: dict) -> str:
    """Generate a deterministic explanation without LLM."""
    claim_id = record.get("claim_id", "Unknown")
    claim_amount = record.get("claim_amount", "N/A")
    
    # Get flags from various possible field names
    flags = record.get("flags", [])
    if not flags:
        flags_str = record.get("flags_list", "") or record.get("anomaly_reasons_str", "")
        if flags_str:
            flags = [f.strip() for f in flags_str.split(",") if f.strip()]
    
    if not flags:
        return f"Claim {claim_id} was flagged for review but no specific anomaly type was recorded. Please review manually."
    
    # Build explanation from flags
    explanations = []
    remediations = []
    
    for flag in flags:
        flag_lower = flag.lower().strip()
        if flag_lower in FLAG_DESCRIPTIONS:
            explanations.append(FLAG_DESCRIPTIONS[flag_lower])
        if flag_lower in FLAG_REMEDIATIONS:
            remediations.append(FLAG_REMEDIATIONS[flag_lower])
    
    # Compose final explanation
    if explanations:
        explanation_text = f"Claim {claim_id} (${claim_amount}) was flagged due to: {'; '.join(explanations)}."
    else:
        explanation_text = f"Claim {claim_id} (${claim_amount}) was flagged for: {', '.join(flags)}."
    
    if remediations:
        remediation_text = f" Recommended action: {remediations[0]}"
    else:
        remediation_text = " Please review this claim manually and verify all details."
    
    return explanation_text + remediation_text


def explain_anomaly(record: dict) -> str:
    """
    Generate an explanation for an anomalous claim.
    
    Uses OpenAI API if OPENAI_API_KEY is set, otherwise falls back to
    deterministic explanation based on flags.
    
    Args:
        record: Dictionary containing claim details and flags
    
    Returns:
        String explanation with remediation suggestion
    """
    # Try OpenAI first if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        prompt = build_prompt(record)
        llm_response = call_openai(prompt)
        if llm_response:
            return llm_response
    
    # Fallback to deterministic explanation
    return generate_fallback_explanation(record)


def explain_batch(records: list, max_records: int = 10) -> list:
    """
    Generate explanations for a batch of anomalous claims.
    
    Args:
        records: List of dictionaries containing claim details
        max_records: Maximum number of records to process
    
    Returns:
        List of explanation strings
    """
    explanations = []
    for i, record in enumerate(records[:max_records]):
        explanation = explain_anomaly(record)
        explanations.append({
            "claim_id": record.get("claim_id", f"record_{i}"),
            "explanation": explanation
        })
    return explanations


if __name__ == "__main__":
    # Test with sample record
    test_record = {
        "claim_id": "CLM00001234",
        "claim_amount": 50000,
        "provider_id": "PRV00001",
        "member_id": "MBR000123",
        "service_date": "2024-01-15",
        "icd_code": "I10",
        "cpt_code": "99214",
        "flags": ["outlier_amount", "zscore_global"]
    }
    
    print("Testing explain_anomaly:")
    print(explain_anomaly(test_record))