import pickle
import os
import sys

try:
    # Adjust path to point to where we moved the data
    path = 'ml/data/processed/model_info.pkl'
    with open(path, 'rb') as f:
        info = pickle.load(f)
        print(f"Features count: {len(info['features'])}")
        print(f"Features list: {info['features']}")
        print(f"Cat features: {info['cat_features']}")
        print(f"Model paths: {info['model_paths']}")
except Exception as e:
    print(f"Error: {e}")
