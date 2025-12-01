import pandas as pd
import numpy as np
import os

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Should be project root
DATA_DIR = os.path.join(BASE_DIR, 'ml', 'data')

def create_directories():
    """Создает необходимые директории"""
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'models', 'catboost_models'), exist_ok=True)

def safe_div(numerator, denominator):
    return (numerator / (denominator + 1e-6)).replace([np.inf, -np.inf, np.nan], 0)

def advanced_fe(df, is_train=False):
    df = df.copy()
    
    # --- Базовая очистка ---
    # Исключаем колонки, которые точно не числа, чтобы не вызвать ошибку при to_numeric
    exclude_cols = ['dt', 'gender', 'adminarea', 'incomeValueCategory', 'id']
    num_cols_to_convert = df.columns.difference(exclude_cols).tolist()
    
    for col in num_cols_to_convert:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999.0)

    # Работа с датой
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m-%d', errors='coerce')
        df['month'] = df['dt'].dt.month.fillna(-1).astype(int)
        df['dayofweek'] = df['dt'].dt.dayofweek.fillna(-1).astype(int)
        # Hack trick: "Время с начала года" или "Порядковый номер месяца" может ловить инфляцию
        df['month_global'] = df['dt'].dt.year * 12 + df['dt'].dt.month 
        df = df.drop(columns=['dt'])

    # --- НОВЫЕ ФИЧИ (Финансовая логика) ---
    
    # 1. Тренды (Сравнение короткого окна с длинным)
    # Если оборот за 3 мес > оборота за 12 мес (в среднем), значит активность растет
    if 'avg_cur_cr_turn' in df.columns and 'turn_cur_cr_avg_act_v2' in df.columns:
        df['trend_turn_cr'] = safe_div(df['avg_cur_cr_turn'], df['turn_cur_cr_avg_act_v2'])
    
    # 2. Расхождения данных (Data Mismatch)
    # Разница между ЗП в профиле (digital profile) и расчетной ЗП по банку
    if 'dp_ils_avg_salary_1y' in df.columns and 'salary_6to12m_avg' in df.columns:
        df['salary_mismatch'] = df['dp_ils_avg_salary_1y'] - df['salary_6to12m_avg']
        df['salary_mismatch_ratio'] = safe_div(df['dp_ils_avg_salary_1y'], df['salary_6to12m_avg'])

    # 3. Чистый доход (Proxy)
    # Зарплата минус предполагаемые платежи (лимиты часто коррелируют с платежами)
    if 'salary_6to12m_avg' in df.columns and 'hdb_bki_total_max_limit' in df.columns:
        df['income_minus_limit'] = df['salary_6to12m_avg'] - (df['hdb_bki_total_max_limit'] * 0.05) # Допустим 5% от лимита - платеж
    
    # 4. Использование кредитных средств
    if 'hdb_bki_active_cc_max_overdue' in df.columns and 'hdb_bki_active_cc_max_limit' in df.columns:
        df['utilization_cc'] = safe_div(df['hdb_bki_active_cc_max_overdue'], df['hdb_bki_active_cc_max_limit'])

    # --- Старые проверенные Ratios ---
    if 'hdb_bki_total_max_limit' in df.columns and 'salary_6to12m_avg' in df.columns:
        df['limit_to_salary'] = safe_div(df['hdb_bki_total_max_limit'], df['salary_6to12m_avg'])
    
    if 'turn_cur_cr_max_v2' in df.columns and 'turn_cur_cr_avg_v2' in df.columns:
        df['turn_volatility'] = safe_div(df['turn_cur_cr_max_v2'], df['turn_cur_cr_avg_v2'])

    # --- Target Log (Only for train) ---
    if is_train and 'target' in df.columns:
        # We keep original target for metrics, but create log target for training
        pass

    # Категории в строки (Strict Type Enforcement)
    # Define potential categorical columns explicitly
    target_cat_cols = ['gender', 'adminarea', 'incomeValueCategory', 'month', 'dayofweek']
    existing_cat_cols = [c for c in target_cat_cols if c in df.columns]
    
    for c in existing_cat_cols:
        # 1. Fill NaNs
        df[c] = df[c].fillna("Missing")
        
        # 2. Handle numeric-like categories (month, dayofweek) to avoid "2024.0"
        try:
            # Try converting to int first to strip decimals if it's a number
            df[c] = df[c].astype(float).astype(int).astype(str)
        except (ValueError, TypeError):
            # If not a number, just convert to string directly
            df[c] = df[c].astype(str)
            
        # 3. Replace explicit string 'nan' or 'Missing' (normalization)
        df.loc[df[c] == 'nan', c] = "Missing"
            
    return df, existing_cat_cols

# Wrapper for backward compatibility if needed, or main execution
def preprocess_data(df):
    """Wrapper needed for backend real-time inference"""
    return advanced_fe(df, is_train=False)

def main():
    print("=== 1. ПРЕДОБРАБОТКА ДАННЫХ (Advanced) ===")
    create_directories()
    
    # Загрузка данных
    print("Загрузка данных...")
    train_path = os.path.join(DATA_DIR, 'raw', 'hackathon_income_train.csv')
    test_path = os.path.join(DATA_DIR, 'raw', 'hackathon_income_test.csv')
    
    train = pd.read_csv(train_path, sep=';', decimal=',', low_memory=False)
    test = pd.read_csv(test_path, sep=';', decimal=',', low_memory=False)
    
    print(f"Размер трейна: {train.shape}")
    print(f"Размер теста: {test.shape}")
    
    # Предобработка
    print("Предобработка данных...")
    train_processed, cat_cols = advanced_fe(train, is_train=True)
    test_processed, _ = advanced_fe(test, is_train=False)
    
    # Добавление весов (заглушка, если нет в raw)
    if 'w' not in train_processed.columns:
        train_processed['w'] = 1.0
        print("⚠️ Используются веса по умолчанию (1.0)")
    
    # Сохранение обработанных данных
    train_processed.to_csv(os.path.join(DATA_DIR, 'processed', 'train_processed.csv'), index=False)
    test_processed.to_csv(os.path.join(DATA_DIR, 'processed', 'test_processed.csv'), index=False)
    
    # Сохранение списка категориальных признаков
    cat_features_df = pd.DataFrame({'feature': cat_cols})
    cat_features_df.to_csv(os.path.join(DATA_DIR, 'processed', 'categorical_features.csv'), index=False)
    
    print("✅ Предобработка завершена!")
    print(f"Категориальные признаки ({len(cat_cols)}): {cat_cols}...") 

if __name__ == "__main__":
    main()