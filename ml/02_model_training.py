import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import gc

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Should be project root
DATA_DIR = os.path.join(BASE_DIR, 'ml', 'data')

# --- Утилиты ---
def weighted_mae(y_true, y_pred, weights):
    return (weights * np.abs(y_true - y_pred)).mean()

def load_processed_data():
    """Загрузка обработанных данных"""
    print("Загрузка обработанных данных...")
    train_path = os.path.join(DATA_DIR, 'processed', 'train_processed.csv')
    test_path = os.path.join(DATA_DIR, 'processed', 'test_processed.csv')
    sub_path = os.path.join(DATA_DIR, 'raw', 'sample_submission.csv')
    cat_path = os.path.join(DATA_DIR, 'processed', 'categorical_features.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sub_path, sep=',')
    
    # Check if categorical_features exists, otherwise infer from dtypes (object)
    if os.path.exists(cat_path):
        cat_features = pd.read_csv(cat_path)['feature'].tolist()
    else:
        cat_features = train.select_dtypes(include=['object']).columns.tolist()
    
    return train, test, submission, cat_features

def train_model():
    print("=== 2. ОБУЧЕНИЕ АНСАМБЛЯ (CatBoost + LightGBM) ===")
    
    # Загрузка данных
    train, test, submission, cat_cols = load_processed_data()
    
    # Подготовка X, y, w
    # Удаляем лишние колонки, если они есть
    drop_cols = ['target', 'w', 'id', 'target_log'] 
    X = train.drop(columns=[c for c in drop_cols if c in train.columns])
    
    # Логарифмируем таргет для обучения (как в твоем новом коде)
    y_log = np.log1p(train['target']) 
    w = train['w']
    
    X_test = test.drop(columns=['id'])
    
    # Выравниваем колонки
    common_cols = [c for c in X.columns if c in X_test.columns]
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    # Обновляем список категориальных признаков (только те, что остались)
    cat_features = [c for c in cat_cols if c in X.columns]
    print(f"Используется {len(cat_features)} категориальных признаков")
    
    # --- ROBUST CLEANUP (Fixing NaNs for CatBoost) ---
    print("Принудительная очистка категориальных признаков от NaN...")
    for col in cat_features:
        # Fill NaNs with "Missing" and convert to string
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)
        
        # Replace any accidental string 'nan' with 'Missing'
        X.loc[X[col] == 'nan', col] = 'Missing'
        X_test.loc[X_test[col] == 'nan', col] = 'Missing'

    # --- Подготовка для LightGBM (Label Encoding) ---
    print("Подготовка данных для LightGBM...")
    X_lgbm = X.copy()
    X_test_lgbm = X_test.copy()
    
    for col in cat_features:
        le = LabelEncoder()
        # Объединяем для фита, чтобы не было unknown label
        full_col = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(full_col)
        X_lgbm[col] = le.transform(X[col].astype(str))
        X_test_lgbm[col] = le.transform(X_test[col].astype(str))

    # --- Параметры моделей ---
    SEED = 42
    N_SPLITS = 5 # Оставим 5 для скорости хакатона (в твоем коде было 10)
    
    lgbm_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 2000, # Чуть меньше для скорости
        'learning_rate': 0.03,
        'num_leaves': 64,
        'max_depth': 10,
        'reg_alpha': 1,
        'reg_lambda': 5,
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    }

    cat_params = {
        'iterations': 2000,
        'learning_rate': 0.059,
        'depth': 8,
        'l2_leaf_reg': 6.9,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': SEED,
        'verbose': 100,
        'allow_writing_files': False,
        'task_type': "CPU"
    }

    # --- Обучение ---
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    cat_preds_test = np.zeros(len(X_test))
    lgbm_preds_test = np.zeros(len(X_test))
    scores = []
    
    models_dir = os.path.join(DATA_DIR, 'models', 'catboost_models')
    os.makedirs(models_dir, exist_ok=True)
    
    saved_catboost_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        print(f"\n--- Fold {fold + 1} ---")
        
        # Данные для CatBoost
        X_tr, y_tr, w_tr = X.iloc[train_idx], y_log.iloc[train_idx], w.iloc[train_idx]
        X_val, y_val, w_val = X.iloc[val_idx], y_log.iloc[val_idx], w.iloc[val_idx]
        
        # Данные для LightGBM
        X_tr_lgbm, X_val_lgbm = X_lgbm.iloc[train_idx], X_lgbm.iloc[val_idx]
        
        # 1. CatBoost
        train_pool = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, weight=w_val, cat_features=cat_features)
        
        model_cb = CatBoostRegressor(**cat_params)
        model_cb.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        
        # Save CatBoost model for API
        model_path = os.path.join(models_dir, f'model_fold_{fold}.cbm')
        model_cb.save_model(model_path)
        saved_catboost_models.append(model_path)
        
        val_pred_cb = np.expm1(model_cb.predict(X_val))
        cat_preds_test += np.expm1(model_cb.predict(X_test)) / N_SPLITS
        
        # 2. LightGBM
        model_lgbm = LGBMRegressor(**lgbm_params)
        model_lgbm.fit(X_tr_lgbm, y_tr, sample_weight=w_tr,
                       eval_set=[(X_val_lgbm, y_val)],
                       eval_sample_weight=[w_val],
                       callbacks=[])
        
        val_pred_lgbm = np.expm1(model_lgbm.predict(X_val_lgbm))
        lgbm_preds_test += np.expm1(model_lgbm.predict(X_test_lgbm)) / N_SPLITS
        
        # Blend
        val_pred_blend = 0.6 * val_pred_cb + 0.4 * val_pred_lgbm
        
        # Score
        y_true_orig = np.expm1(y_val)
        score = weighted_mae(y_true_orig, val_pred_blend, w_val)
        scores.append(score)
        print(f"Fold {fold+1} WMAE: {score:.4f}")
        
        # Cleanup
        del model_cb, model_lgbm, train_pool, val_pool
        gc.collect()

    print(f"\n✅ Средний WMAE: {np.mean(scores):.4f}")
    
    # --- Сохранение результатов ---
    final_pred = 0.6 * cat_preds_test + 0.4 * lgbm_preds_test
    submission['target'] = np.maximum(0, final_pred)
    
    sub_save_path = os.path.join(DATA_DIR, 'processed', 'submission_wmae.csv')
    submission.to_csv(sub_save_path, index=False)
    
    # Save info for API/SHAP
    model_info = {
        'model_paths': saved_catboost_models,
        'features': X.columns.tolist(),
        'cat_features': cat_features,
        'test_ids': test['id'].tolist()
    }
    
    with open(os.path.join(DATA_DIR, 'processed', 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)
        
    print("✅ Обучение завершено! Файл submission_wmae.csv обновлен.")

if __name__ == "__main__":
    train_model()
