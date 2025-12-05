# Abacus - Healthcare Claims Data Quality Pipeline

A comprehensive data quality pipeline for healthcare claims processing, featuring anomaly detection, LLM-powered explanations, and FHIR compatibility.


---

### 1. Streamlit Dashboard (Live Demo)

🔗 **Link**: [https://ramprakash852-abacus-hackathon-srcdashboardapp-up4nsy.streamlit.app/](https://ramprakash852-abacus-hackathon-srcdashboardapp-up4nsy.streamlit.app/)

**Description**:
- This is the **interactive UI** of the platform.
- Users can explore anomaly detection results, data quality insights, KPIs, and AI explanations.
- The dashboard visualizes outputs from the **Gold Layer** and presents insights for analysts and decision-makers.

---

### 2. FastAPI Backend

🔗 **Link**: [https://payerinsightx.onrender.com/docs](https://payerinsightx.onrender.com/docs)

**Description**:
- Backend service exposing REST endpoints such as `/anomalies` and `/claim/{id}/explanation`.
- Enables **programmatic access** to anomaly insights and explanations.
- Demonstrates real-world FDE-level API integration.
- Interactive API documentation available via Swagger UI.

---

### 3. FHIR Resources Viewer

🔗 **Link**: [https://abacus-hackathon-fpkbbcsj6pkdggugbnhfql.streamlit.app/](https://abacus-hackathon-fpkbbcsj6pkdggugbnhfql.streamlit.app/)

*(Also accessible from the main Streamlit Dashboard sidebar)*

**Description**:
- A viewer for **FHIR R4 resources** (Claim, Patient, Practitioner, Encounter).
- Shows how cleaned claims are converted into **FHIR-standard JSON**.
- Highlights interoperability and modern healthcare data standards.

---

## Features

- **Data Ingestion**: CSV to Parquet conversion with schema validation
- **Data Transformation**: Date normalization, string cleaning, code validation
- **Data Quality Rules**: 7 built-in rules for healthcare claims validation
- **Anomaly Detection**: Z-score based outlier detection at multiple levels
- **LLM Explanations**: AI-powered (or deterministic fallback) anomaly explanations
- **FHIR Export**: Convert claims data to FHIR R4 resources
- **Interactive Dashboards**: Streamlit-based visualization tools

## Project Structure

```
Abacus/
├── data/
│   ├── bronze/          # Raw ingested data (Parquet)
│   ├── silver/          # Cleaned/transformed data
│   ├── gold/            # Final outputs (anomalies, clean claims)
│   └── fhir/            # FHIR JSON resources
├── src/
│   ├── ingestion/       # CSV ingestion module
│   ├── transform/       # Data transformation
│   ├── dq/              # Data quality rules engine
│   ├── anomaly/         # Anomaly detection
│   ├── llm/             # LLM explanation module
│   ├── fhir/            # FHIR mapper
│   ├── dashboard/       # Streamlit dashboards
│   └── run_pipeline.py  # Main orchestrator
├── tests/               # Pytest test suite
└── config/              # Configuration files
```

## Quick Start

### Setup Environment

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install pandas numpy pyarrow streamlit openai pytest
```

### Generate Sample Data

```bash
python data/generate_synthetic.py
```

### Run the Pipeline

```bash
python src/run_pipeline.py
```

### Run with FHIR Export

```bash
python src/run_pipeline.py --export-fhir
```

### Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Data Quality Rules

| Rule | Description |
|------|-------------|
| `missing_mandatory` | Missing required fields (claim_id, member_id, provider_id, amount, date) |
| `duplicate_claim` | Duplicate claim IDs detected |
| `invalid_date` | Invalid or future service dates |
| `invalid_amount` | Negative or zero claim amounts |
| `invalid_icd` | Invalid ICD-10 diagnosis code format |
| `invalid_cpt` | Invalid CPT procedure code format |
| `outlier_amount` | Statistical outliers (z-score > 3) |

## FHIR Compatibility Layer

This project can export basic FHIR resources from cleaned data:
- Patient (FHIR Patient)
- Practitioner (FHIR Practitioner)
- Encounter (FHIR Encounter)
- Claim (FHIR Claim)

Run pipeline to export:
```bash
python src/run_pipeline.py --export-fhir
# FHIR json files written to data/fhir/
```

Or run the mapper directly:
```bash
python src/fhir/mapper.py
```

View FHIR resources:
```bash
streamlit run src/dashboard/fhir_view.py
```

## Testing

```bash
pytest -q
```

## LLM Integration

Set `OPENAI_API_KEY` environment variable to enable AI-powered explanations:

```bash
set OPENAI_API_KEY=your-api-key  # Windows
# export OPENAI_API_KEY=your-api-key  # Linux/Mac
```

Without an API key, the system uses deterministic rule-based explanations.

## Pipeline Output

After running the pipeline:

- `data/gold/anomalies.csv` - All flagged anomalies with explanations
- `data/gold/anomalies.parquet` - Same in Parquet format
- `data/gold/claims_clean.parquet` - Clean claims passing all DQ rules
- `data/fhir/` - FHIR R4 JSON resources (if `--export-fhir` flag used)

---

## Live Deployments

### About the Deployment

The dashboard, API, and FHIR viewer are deployed using cloud hosting platforms (Streamlit Cloud, Render). This deployment demonstrates production-readiness through modular architecture, separate serving layers, and pre-generated data outputs. The pipeline is not executed in the cloud; instead, Gold-layer outputs and FHIR files are pre-generated locally and included in the repository for demo purposes.


## License

MIT
