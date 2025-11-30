import pandas as pd
import numpy as np
import os

def create_directories():
    """Создает необходимые директории"""
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models/catboost_models', exist_ok=True)

def preprocess_data(df):
    """Функция предобработки данных"""
    # --- Работа с датами ---
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
        df['month'] = df['dt'].dt.month
        df['year'] = df['dt'].dt.year
        df['day_of_week'] = df['dt'].dt.dayofweek
        df = df.drop(columns=['dt'])
    
    # --- Заполнение пропусков и приведение типов ---
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in cat_features:
        df[col] = df[col].fillna("Missing")
        df[col] = df[col].astype(str)
        
    return df, cat_features

def main():
    print("=== 1. ПРЕДОБРАБОТКА ДАННЫХ ===")
    create_directories()
    
    # Загрузка данных
    print("Загрузка данных...")
    train = pd.read_csv('data/raw/hackathon_income_train.csv', sep=';', decimal=',', low_memory=False)
    test = pd.read_csv('data/raw/hackathon_income_test.csv', sep=';', decimal=',', low_memory=False)
    
    print(f"Размер трейна: {train.shape}")
    print(f"Размер теста: {test.shape}")
    
    # Предобработка
    print("Предобработка данных...")
    train_processed, cat_cols = preprocess_data(train)
    test_processed, _ = preprocess_data(test)
    
    # Добавление весов (заглушка)
    if 'w' not in train_processed.columns:
        train_processed['w'] = 1.0
        print("⚠️ Используются веса по умолчанию (1.0)")
    
    # Сохранение обработанных данных
    train_processed.to_csv('data/processed/train_processed.csv', index=False)
    test_processed.to_csv('data/processed/test_processed.csv', index=False)
    
    # Сохранение списка категориальных признаков
    cat_features_df = pd.DataFrame({'feature': cat_cols})
    cat_features_df.to_csv('data/processed/categorical_features.csv', index=False)
    
    print("✅ Предобработка завершена!")
    print(f"Категориальные признаки ({len(cat_cols)}): {cat_cols[:5]}...")  # Покажем первые 5

if __name__ == "__main__":
    main()