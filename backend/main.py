from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import json
import sys
from catboost import CatBoostRegressor
import shap

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from ml.data_preprocessing import preprocess_data
    from ml.generate_recommendations import generate_recommendations
except ImportError as e:
    import importlib.util
    
    def import_from_path(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    preprocess_module = import_from_path("preprocess_module", os.path.join(project_root, "ml", "01_data_preprocessing.py"))
    recommendations_module = import_from_path("recommendations_module", os.path.join(project_root, "ml", "04_generate_recommendations.py"))
    
    preprocess_data = preprocess_module.preprocess_data
    generate_recommendations = recommendations_module.generate_recommendations

app = FastAPI(title="Alpha Bank Income Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = {
    "submission": None,
    "shap": None,
    "recommendations": None,
    "model": None,
    "explainer": None,
    "defaults": None,
    "test_data": None  
}

def load_resources():
    base_path = os.path.join(project_root, "ml", "data", "processed")
    model_base_path = os.path.join(project_root, "ml", "data", "models", "catboost_models")
    defaults_path = os.path.join(current_dir, "defaults.json")

    print("--- Loading Resources ---")

    try:
        sub_path = os.path.join(base_path, "submission_wmae.csv")
        if os.path.exists(sub_path):
            data_store["submission"] = pd.read_csv(sub_path)
            data_store["submission"]['id'] = data_store["submission"]['id'].astype(str)
            print(f"✅ Submission data: {len(data_store['submission'])}")
        
        shap_path = os.path.join(base_path, "shap_values_frontend.csv")
        if os.path.exists(shap_path):
            data_store["shap"] = pd.read_csv(shap_path)
            data_store["shap"]['id'] = data_store["shap"]['id'].astype(str)
            print(f"✅ SHAP data (static): {len(data_store['shap'])}")

        rec_path = os.path.join(base_path, "client_recommendations.csv")
        if os.path.exists(rec_path):
            data_store["recommendations"] = pd.read_csv(rec_path)
            data_store["recommendations"]['id'] = data_store["recommendations"]['id'].astype(str)
            print(f"✅ Recommendations: {len(data_store['recommendations'])}")
            
        test_path = os.path.join(base_path, "test_processed.csv")
        if os.path.exists(test_path):
            print("⏳ Loading test data for fallback SHAP (this may take a moment)...")
            df_test = pd.read_csv(test_path)
            df_test['id'] = df_test['id'].astype(str)
            data_store["test_data"] = df_test.set_index('id')
            print(f"✅ Test data loaded: {len(data_store['test_data'])} records")
            
    except Exception as e:
        print(f"❌ Error loading static data: {e}")

    try:
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r') as f:
                data_store["defaults"] = json.load(f)
            print(f"✅ Defaults loaded: {len(data_store['defaults'])} features")
        else:
            print(f"❌ Defaults file not found at {defaults_path}")
    except Exception as e:
        print(f"❌ Error loading defaults: {e}")

    try:
        model_path = os.path.join(model_base_path, "model_fold_0.cbm")
        if os.path.exists(model_path):
            model = CatBoostRegressor()
            model.load_model(model_path)
            data_store["model"] = model
            print(f"✅ CatBoost model loaded from {model_path}")
            
            print("⏳ Initializing SHAP explainer (this might take a few seconds)...")
            data_store["explainer"] = shap.TreeExplainer(model)
            print("✅ SHAP Explainer initialized")
            
        else:
            print(f"❌ Model file not found at {model_path}")
            if os.path.exists(model_base_path):
                 files = [f for f in os.listdir(model_base_path) if f.endswith('.cbm')]
                 if files:
                     model_path = os.path.join(model_base_path, files[0])
                     model = CatBoostRegressor()
                     model.load_model(model_path)
                     data_store["model"] = model
                     print(f"✅ CatBoost model loaded from fallback {model_path}")
                     
                     print("⏳ Initializing SHAP explainer...")
                     data_store["explainer"] = shap.TreeExplainer(model)
                     print("✅ SHAP Explainer initialized")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

load_resources()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Alpha Bank Prediction API", "model_loaded": data_store["model"] is not None}

@app.get("/api/features")
def get_features():
    """
    Returns the list of all features and their default values.
    Used to build the dynamic form on the frontend.
    """
    if data_store["defaults"] is None:
        raise HTTPException(status_code=503, detail="Defaults not available")
    return data_store["defaults"]

@app.get("/api/clients/search")
def search_clients(q: str = ""):
    """
    Search for clients by ID prefix.
    Returns top 10 matches.
    Used for autocomplete.
    """
    if data_store["submission"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    df = data_store["submission"]
    
    if q:
        mask = df['id'].astype(str).str.startswith(q)
        results = df[mask].head(10)
    else:
        results = df.head(10)
        
    return results[['id', 'target']].rename(columns={'target': 'predicted_income'}).to_dict(orient='records')

@app.get("/api/client/{client_id}")
def get_client_data(client_id: str):
    if data_store["submission"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    client_sub = data_store["submission"][data_store["submission"]['id'] == client_id]
    
    if client_sub.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    
    result = {
        "id": client_id,
        "submission": client_sub.to_dict(orient='records')[0],
        "shap": None,
        "recommendations": None
    }
    
    shap_found = False
    
    if data_store["shap"] is not None:
        client_shap = data_store["shap"][data_store["shap"]['id'] == client_id]
        if not client_shap.empty:
            shap_dict = client_shap.to_dict(orient='records')[0]
            result["shap"] = {k: (0 if pd.isna(v) else v) for k, v in shap_dict.items()}
            print(f"✅ Found static SHAP data for {client_id}")
            shap_found = True

    if not shap_found and data_store["test_data"] is not None and data_store["explainer"] is not None:
        try:
            if client_id in data_store["test_data"].index:
                print(f"⏳ Calculating real-time SHAP for {client_id}...")
                client_features = data_store["test_data"].loc[client_id]
                
                # Convert to DataFrame (1 row)
                df_features = pd.DataFrame([client_features])
                
                # --- SAFETY CHECK (Same as in predict_income) ---
                # 1. Reorder columns
                if hasattr(data_store["model"], "feature_names_"):
                    expected_cols = data_store["model"].feature_names_
                    for c in expected_cols:
                        if c not in df_features.columns:
                            df_features[c] = 0
                    df_features = df_features[expected_cols]

                # 2. Clean Categoricals
                if hasattr(data_store["model"], "get_cat_feature_indices"):
                    cat_indices = data_store["model"].get_cat_feature_indices()
                    all_columns = df_features.columns
                    for idx in cat_indices:
                        if idx < len(all_columns):
                            col_name = all_columns[idx]
                            try:
                                val = df_features[col_name].iloc[0]
                                if pd.api.types.is_number(val):
                                    df_features[col_name] = str(int(val))
                                else:
                                    df_features[col_name] = str(val)
                            except:
                                df_features[col_name] = str(df_features[col_name].iloc[0])
                
                df_features = df_features.fillna(0)
                
                # Calculate SHAP
                shap_values = data_store["explainer"].shap_values(df_features)
                
                # Map to dict
                shap_dict = dict(zip(df_features.columns, shap_values[0]))
                result["shap"] = shap_dict
                print(f"✅ Calculated real-time SHAP for {client_id}")
                shap_found = True
            else:
                print(f"⚠️ Client {client_id} not found in test data for fallback")
        except Exception as e:
            print(f"❌ Error calculating real-time SHAP: {e}")

    if not shap_found:
        print(f"⚠️ No SHAP data available for {client_id}")

    rec_found = False

    if data_store["recommendations"] is not None:
        client_rec = data_store["recommendations"][data_store["recommendations"]['id'] == client_id]
        if not client_rec.empty:
            result["recommendations"] = client_rec.to_dict(orient='records')[0]
            print(f"✅ Found static Recommendations for {client_id}")
            rec_found = True

    if not rec_found:
        try:
            print(f"⏳ Generating real-time recommendations for {client_id}...")
            predicted_income = result["submission"].get("target", 0)
            
            top_features = []
            if result["shap"]:
                top_features = sorted(
                    result["shap"].items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
            
            recs_list = generate_recommendations(predicted_income, top_features, {})
            
            result["recommendations"] = {
                "id": client_id,
                "recommendations": ' | '.join(recs_list)
            }
            print(f"✅ Generated real-time recommendations for {client_id}")
            rec_found = True
            
        except Exception as e:
            print(f"❌ Error generating real-time recommendations: {e}")

    if not rec_found:
         print(f"⚠️ No Recommendations available for {client_id}")

    return result

@app.post("/api/predict")
def predict_income(features: dict = Body(...)):
    """
    Real-time prediction using User's Python logic + Real-time SHAP.
    """
    if data_store["model"] is None:
        raise HTTPException(status_code=503, detail="ML Model is not available")
    
    if data_store["defaults"] is None:
        raise HTTPException(status_code=503, detail="Default features are not available")

    try:
        input_data = data_store["defaults"].copy()
        for key, value in features.items():
            if key in input_data:
                try:
                    if isinstance(input_data[key], (int, float)):
                         input_data[key] = float(value)
                    else:
                         input_data[key] = str(value)
                except:
                    pass 

        df_input = pd.DataFrame([input_data])
        
        # 3. Preprocess
        df_processed, _ = preprocess_data(df_input)
        
        # --- SAFETY: Reorder columns to match model expectation ---
        if hasattr(data_store["model"], "feature_names_"):
            expected_cols = data_store["model"].feature_names_
            # Add missing cols with 0
            for c in expected_cols:
                if c not in df_processed.columns:
                    df_processed[c] = 0
            # Reorder
            df_processed = df_processed[expected_cols]

        # --- FINAL SAFETY CHECK (API Layer) ---
        # Ensure all categorical features expected by CatBoost are actually strings
        if hasattr(data_store["model"], "get_cat_feature_indices"):
            cat_indices = data_store["model"].get_cat_feature_indices()
            # Map indices to column names
            all_columns = df_processed.columns
            for idx in cat_indices:
                if idx < len(all_columns):
                    col_name = all_columns[idx]
                    # Force to string, handling floats like 2024.0 -> "2024"
                    try:
                        val = df_processed[col_name].iloc[0]
                        if pd.api.types.is_number(val):
                            df_processed[col_name] = str(int(val))
                        else:
                            df_processed[col_name] = str(val)
                    except:
                        df_processed[col_name] = str(df_processed[col_name].iloc[0])
        
        # Ensure no NaNs in numerical columns (fill with 0 or median from defaults if possible, here 0 is safe)
        df_processed = df_processed.fillna(0)

        # 4. Predict
        prediction = data_store["model"].predict(df_processed)[0]
        predicted_value = float(np.expm1(prediction))

        # 5. Calculate SHAP Values
        shap_dict = None
        if data_store["explainer"]:
            shap_values = data_store["explainer"].shap_values(df_processed)
            shap_dict = dict(zip(df_processed.columns, shap_values[0]))

        shap_tuples = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True) if shap_dict else []
        
        recs_list = generate_recommendations(predicted_value, shap_tuples, shap_dict)
        recommendation_text = ' | '.join(recs_list)

        return {
            "predicted_income": predicted_value,
            "recommendations": recommendation_text,
            "shap": shap_dict  
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
