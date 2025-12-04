"""
Simple Streamlit view to browse a few exported FHIR JSON resources

- Load files under data/fhir
- Show a table of files
- When a file is selected show prettified JSON and selected key fields
"""
import streamlit as st
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FHIR Resources Viewer",
    page_icon="🏥",
    layout="wide"
)


def get_fhir_files(base_dir: str = "data/fhir") -> dict:
    """Get all FHIR JSON files organized by resource type."""
    base_path = Path(base_dir)
    files_by_type = {}
    
    if not base_path.exists():
        return files_by_type
    
    # Check subdirectories (Patient, Practitioner, Claim, Encounter)
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            resource_type = subdir.name
            json_files = sorted(subdir.glob("*.json"))
            if json_files:
                files_by_type[resource_type] = json_files
    
    # Also check root directory for any JSON files
    root_files = list(base_path.glob("*.json"))
    if root_files:
        files_by_type["Other"] = sorted(root_files)
    
    return files_by_type


def display_patient(data: dict):
    """Display Patient resource details."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        st.write(f"**ID:** {data.get('id', 'N/A')}")
        
        # Name
        names = data.get("name", [])
        if names:
            name = names[0]
            full_name = f"{' '.join(name.get('given', []))} {name.get('family', '')}"
            st.write(f"**Name:** {full_name}")
        
        st.write(f"**Gender:** {data.get('gender', 'N/A')}")
        st.write(f"**Birth Date:** {data.get('birthDate', 'N/A')}")
        st.write(f"**Active:** {data.get('active', 'N/A')}")
    
    with col2:
        st.subheader("Identifiers")
        for ident in data.get("identifier", []):
            st.write(f"- **System:** {ident.get('system', 'N/A')}")
            st.write(f"  **Value:** {ident.get('value', 'N/A')}")


def display_practitioner(data: dict):
    """Display Practitioner resource details."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Practitioner Information")
        st.write(f"**ID:** {data.get('id', 'N/A')}")
        
        # Name
        names = data.get("name", [])
        if names:
            name = names[0]
            st.write(f"**Name:** {name.get('text', 'N/A')}")
        
        st.write(f"**Active:** {data.get('active', 'N/A')}")
        
        # Address
        addresses = data.get("address", [])
        if addresses:
            addr = addresses[0]
            st.write(f"**State:** {addr.get('state', 'N/A')}")
    
    with col2:
        st.subheader("Qualifications")
        for qual in data.get("qualification", []):
            code = qual.get("code", {})
            st.write(f"- {code.get('text', 'N/A')}")
        
        st.subheader("Identifiers")
        for ident in data.get("identifier", []):
            system = ident.get('system', 'N/A')
            if 'npi' in system.lower():
                st.write(f"**NPI:** {ident.get('value', 'N/A')}")
            else:
                st.write(f"**ID:** {ident.get('value', 'N/A')}")


def display_claim(data: dict):
    """Display Claim resource details."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claim Information")
        st.write(f"**ID:** {data.get('id', 'N/A')}")
        st.write(f"**Status:** {data.get('status', 'N/A')}")
        st.write(f"**Use:** {data.get('use', 'N/A')}")
        st.write(f"**Created:** {data.get('created', 'N/A')}")
        
        # Total
        total = data.get("total", {})
        if total:
            st.write(f"**Total:** ${total.get('value', 0):,.2f} {total.get('currency', 'USD')}")
        
        # References
        patient = data.get("patient", {})
        if patient:
            st.write(f"**Patient:** {patient.get('reference', 'N/A')}")
        
        provider = data.get("provider", {})
        if provider:
            st.write(f"**Provider:** {provider.get('reference', 'N/A')}")
    
    with col2:
        st.subheader("Diagnosis")
        for diag in data.get("diagnosis", []):
            code_concept = diag.get("diagnosisCodeableConcept", {})
            codings = code_concept.get("coding", [])
            for coding in codings:
                st.write(f"- **ICD-10:** {coding.get('code', 'N/A')}")
        
        st.subheader("Procedures")
        for proc in data.get("procedure", []):
            code_concept = proc.get("procedureCodeableConcept", {})
            codings = code_concept.get("coding", [])
            for coding in codings:
                st.write(f"- **CPT:** {coding.get('code', 'N/A')}")


def display_encounter(data: dict):
    """Display Encounter resource details."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Encounter Information")
        st.write(f"**ID:** {data.get('id', 'N/A')}")
        st.write(f"**Status:** {data.get('status', 'N/A')}")
        
        # Class
        enc_class = data.get("class", {})
        if enc_class:
            st.write(f"**Class:** {enc_class.get('display', enc_class.get('code', 'N/A'))}")
        
        # Period
        period = data.get("period", {})
        if period:
            st.write(f"**Start:** {period.get('start', 'N/A')}")
            st.write(f"**End:** {period.get('end', 'N/A')}")
    
    with col2:
        st.subheader("References")
        subject = data.get("subject", {})
        if subject:
            st.write(f"**Patient:** {subject.get('reference', 'N/A')}")
        
        for participant in data.get("participant", []):
            individual = participant.get("individual", {})
            if individual:
                st.write(f"**Practitioner:** {individual.get('reference', 'N/A')}")


def app():
    st.title("🏥 FHIR Resources Viewer")
    st.markdown("Browse exported FHIR R4 resources from the Abacus pipeline")
    
    # Get all FHIR files
    files_by_type = get_fhir_files()
    
    if not files_by_type:
        st.warning("No FHIR resources found. Run the pipeline with `--export-fhir` flag first:")
        st.code("python src/run_pipeline.py --export-fhir")
        st.info("Or run the FHIR mapper directly:")
        st.code("python src/fhir/mapper.py")
        return
    
    # Sidebar for navigation
    st.sidebar.header("📁 Resource Browser")
    
    # Resource type selection
    resource_types = list(files_by_type.keys())
    selected_type = st.sidebar.selectbox("Resource Type", resource_types)
    
    # File selection
    if selected_type:
        files = files_by_type[selected_type]
        file_names = [f.name for f in files]
        
        # Show count
        st.sidebar.write(f"Found {len(files)} {selected_type} resources")
        
        selected_file = st.sidebar.selectbox(f"Select {selected_type}", file_names)
        
        if selected_file:
            # Find full path
            file_path = next(f for f in files if f.name == selected_file)
            
            # Load and display
            with open(file_path) as fh:
                data = json.load(fh)
            
            # Display header
            st.header(f"{data.get('resourceType', 'Unknown')} Resource")
            st.caption(f"File: {file_path}")
            
            # Display formatted details based on resource type
            resource_type = data.get("resourceType")
            
            if resource_type == "Patient":
                display_patient(data)
            elif resource_type == "Practitioner":
                display_practitioner(data)
            elif resource_type == "Claim":
                display_claim(data)
            elif resource_type == "Encounter":
                display_encounter(data)
            else:
                st.write(f"**Resource Type:** {resource_type}")
            
            # Show raw JSON in expander
            with st.expander("📄 View Raw JSON"):
                st.json(data)
    
    # Summary statistics
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Summary")
    for rtype, files in files_by_type.items():
        st.sidebar.write(f"**{rtype}:** {len(files)} resources")


if __name__ == "__main__":
    app()
