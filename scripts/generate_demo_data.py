"""
RetailPulse - Generate Deterministic Synthetic Demo Data Script

Generates at least 24 months of e-commerce transactions with repeat buyers,
cancellations, seasonal patterns, missing customer IDs, and intentional anomalies.
"""

import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import generate_synthetic_demo_data, get_demo_dataset_path

def main():
    dest_path = get_demo_dataset_path()
    print(f"Generating deterministic synthetic demo dataset (seed=42) at: {dest_path}...")
    df = generate_synthetic_demo_data(seed=42)
    df.to_csv(dest_path, index=False)
    print(f"Done! Generated {len(df):,} transaction rows saved to {dest_path}")

if __name__ == "__main__":
    main()
