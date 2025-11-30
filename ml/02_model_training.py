import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import os
import pickle

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Should be project root
DATA_DIR = os.path.join(BASE_DIR, 'ml', 'data')

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
    cat_features = pd.read_csv(cat_path)['feature'].tolist()
    
    return train, test, submission, cat_features

def train_model():
    print("=== 2. ОБУЧЕНИЕ МОДЕЛИ ===")
    
    # Загрузка данных
    train, test, submission, cat_cols = load_processed_data()
    
    # Подготовка features
    X = train.drop(columns=['target', 'w', 'id'])
    y = train['target']
    w = train['w']
    
    X_test = test.drop(columns=['id'])
    
    # Убедимся, что колонки совпадают
    common_cols = [c for c in X.columns if c in X_test.columns]
    X = X[common_cols]
    X_test = X_test[common_cols]
    
    # Обновляем список категориальных признаков
    cat_features = [c for c in cat_cols if c in X.columns]
    print(f"Используется {len(cat_features)} категориальных признаков")
    
    # Обучение модели
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    # Ensure model directory exists
    models_dir = os.path.join(DATA_DIR, 'models', 'catboost_models')
    os.makedirs(models_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold + 1} ---")
        
        X_train, y_train, w_train = X.iloc[train_idx], y.iloc[train_idx], w.iloc[train_idx]
        X_val, y_val, w_val = X.iloc[val_idx], y.iloc[val_idx], w.iloc[val_idx]
        
        # Создаем Pool для CatBoost
        train_pool = Pool(data=X_train, label=y_train, weight=w_train, cat_features=cat_features)
        val_pool = Pool(data=X_val, label=y_val, weight=w_val, cat_features=cat_features)
        
        model = CatBoostRegressor(
            iterations=1000,  # Уменьшил для скорости, можно вернуть 2000
            learning_rate=0.05,
            depth=6,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=42 + fold,
            verbose=100,  # Уменьшил вербозность
            early_stopping_rounds=50,
            task_type="CPU"
        )
        
        print("Обучение модели...")
        model.fit(train_pool, eval_set=val_pool)
        
        # Сохраняем модель
        model_path = os.path.join(models_dir, f'model_fold_{fold}.cbm')
        model.save_model(model_path)
        models.append(model_path)
        
        # Предсказание
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        
        # Предсказание на тест
        test_preds += model.predict(X_test) / n_folds
        
        # Расчет метрики на фолде
        w_val_sum = np.sum(w_val)
        if w_val_sum > 0:
            fold_wmae = np.sum(w_val * np.abs(y_val - val_pred)) / w_val_sum
            print(f"Fold {fold + 1} WMAE: {fold_wmae:.4f}")
    
    # Итоговая метрика
    w_sum = np.sum(w)
    if w_sum > 0:
        total_wmae = np.sum(w * np.abs(y - oof_preds)) / w_sum
        print(f"\n✅ Итоговый OOF WMAE: {total_wmae:.4f}")
    
    # Сохранение предсказаний
    submission['target'] = test_preds
    submission.to_csv(os.path.join(DATA_DIR, 'processed', 'submission_wmae.csv'), index=False)
    
    # Сохранение информации о моделях
    model_info = {
        'model_paths': models,
        'features': X.columns.tolist(),
        'cat_features': cat_features,
        'test_ids': test['id'].tolist()
    }
    
    with open(os.path.join(DATA_DIR, 'processed', 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)
    
    print("✅ Обучение завершено!")
    print(f"Сохранено {len(models)} моделей")
    print(f"Файл submission_wmae.csv создан")

if __name__ == "__main__":
    train_model()