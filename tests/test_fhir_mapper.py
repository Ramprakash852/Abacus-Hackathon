def test_map_patient_and_claim():
    import pandas as pd
    from src.fhir.mapper import map_patient_to_fhir, map_claim_to_fhir
    member = pd.Series({"member_id":"m1","name":"John Doe","dob":"1980-01-01"})
    provider = pd.Series({"provider_id":"pr1","name":"Dr Smith","specialty":"Cardiology"})
    claim = pd.Series({"claim_id":"c1","member_id":"m1","provider_id":"pr1","claim_amount":250.0,"service_date":"2025-01-10","icd_code":"I10","cpt_code":"99213","claim_status":"paid"})
    p = map_patient_to_fhir(member)
    c = map_claim_to_fhir(claim, member, provider)
    assert p["resourceType"] == "Patient"
    assert "id" in p
    assert c["resourceType"] == "Claim"
    assert "patient" in c  # FHIR Claim uses 'patient' not 'subject'
    assert c["patient"]["reference"] == "Patient/m1"
    assert "total" in c
    assert c["total"]["value"] == 250.0
