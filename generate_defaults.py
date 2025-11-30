import pandas as pd
import json
import numpy as np

def generate_defaults():
    print("Loading data...")
    # Read only a sample if file is huge, but for defaults we want accuracy. 
    # If file is > 1GB this might be slow, but let's assume it fits in memory for now or use a sample.
    # Using 'nrows' to be safe if it's massive, but ideally we want full stats.
    try:
        df = pd.read_csv('ml/data/processed/train_processed.csv', nrows=10000)
    except FileNotFoundError:
        print("Train file not found, trying test...")
        df = pd.read_csv('ml/data/processed/test_processed.csv', nrows=10000)

    defaults = {}
    
    print("Calculating defaults...")
    for col in df.columns:
        if col in ['id', 'target', 'w']:
            continue
            
        if df[col].dtype == 'object':
            # Mode for categorical
            val = df[col].mode().iloc[0]
        else:
            # Median for numerical (more robust to outliers)
            val = df[col].median()
            if pd.isna(val):
                val = 0
        
        # Convert numpy types to python native for JSON serialization
        if isinstance(val, (np.int64, np.int32)):
            val = int(val)
        elif isinstance(val, (np.float64, np.float32)):
            val = float(val)
            
        defaults[col] = val

    print(f"Generated defaults for {len(defaults)} features")
    
    with open('backend/defaults.json', 'w') as f:
        json.dump(defaults, f, indent=2)
    print("Saved to backend/defaults.json")

if __name__ == "__main__":
    generate_defaults()