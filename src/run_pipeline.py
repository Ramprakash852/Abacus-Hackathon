import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingest import run_ingest
from src.transform.transform import run_transform
from src.dq.rules import run_dq
from src.anomaly.detect import detect_anomalies
from src.llm.explain import explain_anomaly


def main(explain_top_n: int = 20):
    """
    Run the full data quality pipeline.
    
    Args:
        explain_top_n: Number of top anomalies to generate explanations for
    """
    print("=" * 60)
    print("ABACUS DATA QUALITY PIPELINE")
    print("=" * 60)
    
    # Step 1: Ingest
    print("\n[1/5] INGESTION")
    print("-" * 40)
    ingest_results = run_ingest()
    
    # Step 2: Transform
    print("\n[2/5] TRANSFORMATION")
    print("-" * 40)
    transform_results = run_transform()
    
    # Step 3: Data Quality Rules
    print("\n[3/5] DATA QUALITY RULES")
    print("-" * 40)
    dq_anomalies = run_dq()
    
    # Step 4: Anomaly Detection
    print("\n[4/5] ANOMALY DETECTION")
    print("-" * 40)
    anomalies = detect_anomalies(dq_anomalies)
    
    # Step 5: Generate Explanations
    print("\n[5/5] GENERATING EXPLANATIONS")
    print("-" * 40)
    
    if anomalies is not None and len(anomalies) > 0:
        # Sort by number of flags (most problematic first)
        if "num_flags" in anomalies.columns:
            anomalies_sorted = anomalies.sort_values("num_flags", ascending=False)
        else:
            anomalies_sorted = anomalies
        
        # Generate explanations for top N anomalies
        print(f"Generating explanations for top {explain_top_n} anomalies...")
        explanations = []
        
        for idx, row in anomalies_sorted.head(explain_top_n).iterrows():
            record = row.to_dict()
            explanation = explain_anomaly(record)
            explanations.append(explanation)
        
        # Add explanation column (empty for those beyond top N)
        anomalies["explanation"] = ""
        top_indices = anomalies_sorted.head(explain_top_n).index
        for i, idx in enumerate(top_indices):
            anomalies.loc[idx, "explanation"] = explanations[i]
        
        print(f"  Generated {len(explanations)} explanations")
        
        # Write final outputs
        output_dir = Path("data/gold")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_file = output_dir / "anomalies.parquet"
        csv_file = output_dir / "anomalies.csv"
        
        anomalies.to_parquet(parquet_file, index=False)
        anomalies.to_csv(csv_file, index=False)
        
        print(f"\nFinal outputs written to:")
        print(f"  {parquet_file}")
        print(f"  {csv_file}")
    
    # Print Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    if anomalies is not None:
        total_claims = len(transform_results.get("claims", [])) if transform_results else 0
        total_anomalies = len(anomalies)
        
        print(f"\nTotal claims processed: {total_claims}")
        print(f"Total anomalies found: {total_anomalies}")
        
        if total_claims > 0:
            print(f"Anomaly rate: {total_anomalies/total_claims*100:.1f}%")
        
        # Top anomaly reasons
        if "anomaly_reasons" in anomalies.columns:
            all_reasons = []
            for reasons in anomalies["anomaly_reasons"]:
                if isinstance(reasons, list):
                    all_reasons.extend(reasons)
            
            if all_reasons:
                import pandas as pd
                reason_counts = pd.Series(all_reasons).value_counts().head(10)
                print("\nTop Anomaly Reasons:")
                for reason, count in reason_counts.items():
                    print(f"  {reason}: {count}")
        
        # Sample explanations
        print("\n" + "-" * 40)
        print("Sample Explanations (Top 3):")
        print("-" * 40)
        
        explained = anomalies[anomalies["explanation"] != ""].head(3)
        for _, row in explained.iterrows():
            print(f"\nClaim {row['claim_id']}:")
            print(f"  {row['explanation']}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return anomalies


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Run the Abacus DQ Pipeline")
    parser.add_argument("--explain-top", type=int, default=20, 
                        help="Number of top anomalies to explain (default: 20)")
    parser.add_argument("--export-fhir", action="store_true",
                        help="Export FHIR resources after pipeline completes")
    
    args = parser.parse_args()
    main(explain_top_n=args.explain_top)
    
    # FHIR export (optional)
    if args.export_fhir:
        try:
            from src.fhir.mapper import export_fhir_resources
            # load silver tables
            claims = pd.read_parquet("data/silver/claims.parquet")
            providers = pd.read_parquet("data/silver/providers.parquet")
            members = pd.read_parquet("data/silver/members.parquet")
            print("\nExporting FHIR resources to data/fhir ...")
            export_fhir_resources(claims, providers, members, out_dir="data/fhir", max_claims=100)
            print("FHIR export complete: data/fhir/")
        except Exception as e:
            print("FHIR export skipped due to error:", str(e))
