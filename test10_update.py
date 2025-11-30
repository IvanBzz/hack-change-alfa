import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ---
N_SPLITS = 10  # Увеличим до 10 для макс. надежности
SEED = 42

# --- 1. Загрузка данных ---
print("Загрузка данных...")
train_raw = pd.read_csv('data/raw/hackathon_income_train.csv', sep=';', decimal=',', low_memory=False)
test_raw = pd.read_csv('data/raw/hackathon_income_test.csv', sep=';', decimal=',', low_memory=False)
submission = pd.read_csv('data/raw/sample_submission.csv')

# --- 2. Утилиты ---
def safe_div(numerator, denominator):
    return (numerator / (denominator + 1e-6)).replace([np.inf, -np.inf, np.nan], 0)

def weighted_mae(y_true, y_pred, weights):
    return (weights * np.abs(y_true - y_pred)).mean()

# --- 3. Feature Engineering ---
def advanced_fe(df, is_train=False):
    df = df.copy()
    
    # --- Базовая очистка (как у вас) ---
    num_cols_to_convert = df.columns.difference(['dt', 'gender', 'adminarea', 'incomeValueCategory']).tolist()
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
    df['trend_turn_cr'] = safe_div(df['avg_cur_cr_turn'], df['turn_cur_cr_avg_act_v2'])
    
    # 2. Расхождения данных (Data Mismatch)
    # Разница между ЗП в профиле (digital profile) и расчетной ЗП по банку
    if 'dp_ils_avg_salary_1y' in df.columns and 'salary_6to12m_avg' in df.columns:
        df['salary_mismatch'] = df['dp_ils_avg_salary_1y'] - df['salary_6to12m_avg']
        df['salary_mismatch_ratio'] = safe_div(df['dp_ils_avg_salary_1y'], df['salary_6to12m_avg'])

    # 3. Чистый доход (Proxy)
    # Зарплата минус предполагаемые платежи (лимиты часто коррелируют с платежами)
    df['income_minus_limit'] = df['salary_6to12m_avg'] - (df['hdb_bki_total_max_limit'] * 0.05) # Допустим 5% от лимита - платеж
    
    # 4. Использование кредитных средств
    df['utilization_cc'] = safe_div(df['hdb_bki_active_cc_max_overdue'], df['hdb_bki_active_cc_max_limit'])

    # --- Старые проверенные Ratios ---
    df['limit_to_salary'] = safe_div(df['hdb_bki_total_max_limit'], df['salary_6to12m_avg'])
    df['turn_volatility'] = safe_div(df['turn_cur_cr_max_v2'], df['turn_cur_cr_avg_v2'])

    # --- Target Log ---
    if is_train:
        df['target_log'] = np.log1p(df['target'])

    # Категории в строки
    cat_cols = ['gender', 'adminarea', 'incomeValueCategory', 'month', 'dayofweek']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("Missing")
            
    return df, cat_cols

# Применяем FE
train_df, cat_features = advanced_fe(train_raw, is_train=True)
test_df, _ = advanced_fe(test_raw, is_train=False)

# Подготовка X, y, w
X = train_df.drop(columns=['target', 'target_log', 'w', 'id'])
y_log = train_df['target_log']
w = train_df['w']
X_test = test_df.drop(columns=['id'])

# Выравниваем колонки
common = [c for c in X.columns if c in X_test.columns]
X = X[common]
X_test = X_test[common]

# Label Encoding для LightGBM (он не ест строки напрямую так хорошо, как CatBoost)
X_lgbm = X.copy()
X_test_lgbm = X_test.copy()
lbl_encoders = {}
for col in cat_features:
    if col in X.columns:
        le = LabelEncoder()
        # Объединяем для фита, чтобы не было unknown label
        full_col = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(full_col)
        X_lgbm[col] = le.transform(X[col].astype(str))
        X_test_lgbm[col] = le.transform(X_test[col].astype(str))
        
print(f"Features: {X.shape[1]}")

# --- 4. ОБУЧЕНИЕ АНСАМБЛЯ ---

# Контейнеры для предсказаний
cat_preds = np.zeros(len(X_test))
lgbm_preds = np.zeros(len(X_test))
scores = []

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Параметры LightGBM (быстрые, но мощные)
lgbm_params = {
    'objective': 'regression_l1', # L1 = MAE
    'metric': 'mae',
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'num_leaves': 64,
    'max_depth': 10,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
}

# Ваши лучшие параметры CatBoost
cat_params = {
    'iterations': 4000, # Чуть меньше, т.к. 10 фолдов
    'learning_rate': 0.059,
    'depth': 8,
    'l2_leaf_reg': 6.9,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed': SEED,
    'verbose': 0,
    'allow_writing_files': False
}

print("\n--- Запуск Ансамбля (CatBoost + LightGBM) ---")

feature_importance_df = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
    # Данные
    X_tr, y_tr, w_tr = X.iloc[train_idx], y_log.iloc[train_idx], w.iloc[train_idx]
    X_val, y_val, w_val = X.iloc[val_idx], y_log.iloc[val_idx], w.iloc[val_idx]
    
    # Данные для LightGBM (Label Encoded)
    X_tr_lgbm, X_val_lgbm = X_lgbm.iloc[train_idx], X_lgbm.iloc[val_idx]
    
    # --- MODEL 1: CatBoost ---
    cb_pool_tr = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_features)
    cb_pool_val = Pool(X_val, y_val, weight=w_val, cat_features=cat_features)
    
    model_cb = CatBoostRegressor(**cat_params)
    model_cb.fit(cb_pool_tr, eval_set=cb_pool_val, early_stopping_rounds=150)
    
    val_pred_cb = np.expm1(model_cb.predict(X_val))
    test_pred_cb = np.expm1(model_cb.predict(X_test))
    
    # Сохраняем важность признаков (для 1-го фолда)
    if fold == 0:
        fi = model_cb.get_feature_importance(prettified=True)
        feature_importance_df = fi
    
    # --- MODEL 2: LightGBM ---
    # LightGBM требует веса массивом
    model_lgbm = LGBMRegressor(**lgbm_params)
    model_lgbm.fit(X_tr_lgbm, y_tr, sample_weight=w_tr, 
                   eval_set=[(X_val_lgbm, y_val)], 
                   eval_sample_weight=[w_val],
                   callbacks=[]) # Early stopping встроен через параметры, если нужно
    
    val_pred_lgbm = np.expm1(model_lgbm.predict(X_val_lgbm))
    test_pred_lgbm = np.expm1(model_lgbm.predict(X_test_lgbm))
    
    # --- BLENDING (Смешивание) ---
    # Обычно CatBoost на таких данных чуть лучше, даем ему 60%, LGBM 40%
    # Можно подобрать веса, но 0.5/0.5 или 0.6/0.4 - надежный старт
    val_pred_blend = 0.6 * val_pred_cb + 0.4 * val_pred_lgbm
    
    # Оценка на оригинальном масштабе
    y_true_orig = np.expm1(y_val)
    score = weighted_mae(y_true_orig, val_pred_blend, w_val)
    scores.append(score)
    
    print(f"Fold {fold+1} | Cat: {weighted_mae(y_true_orig, val_pred_cb, w_val):.0f} | LGBM: {weighted_mae(y_true_orig, val_pred_lgbm, w_val):.0f} | Blend: {score:.0f}")
    
    # Накапливаем предсказания для теста
    cat_preds += test_pred_cb / N_SPLITS
    lgbm_preds += test_pred_lgbm / N_SPLITS
    
    # Чистка памяти
    del model_cb, model_lgbm, cb_pool_tr, cb_pool_val
    gc.collect()

print(f"\nСредний WMAE на CV: {np.mean(scores):.2f}")

# --- 5. Анализ Важности ---
print("\nТОП-10 Признаков (CatBoost):")
print(feature_importance_df.head(10))

# --- 6. Финальный Сабмит ---
final_pred = 0.6 * cat_preds + 0.4 * lgbm_preds
submission['target'] = np.maximum(0, final_pred)
submission.to_csv('data/processed/submission_ensemble_final.csv', index=False)
print("Файл submission_ensemble_final.csv сохранен.")