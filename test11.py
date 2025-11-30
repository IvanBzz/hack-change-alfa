import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ---
N_SPLITS = 10  # 10 фолдов для максимальной точности
SEED = 42

# --- ПАРАМЕТРЫ МОДЕЛЕЙ ---
# Ваши лучшие параметры для CatBoost
CAT_PARAMS = {
    'iterations': 5000,
    'learning_rate': 0.0592058,
    'depth': 8,
    'l2_leaf_reg': 6.93184,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed': SEED,
    'verbose': 500,
    'allow_writing_files': False,
    'thread_count': -1
}

# Параметры LightGBM (подобраны под MAE)
LGBM_PARAMS = {
    'objective': 'regression_l1', # L1 = MAE
    'metric': 'mae',
    'n_estimators': 4000,
    'learning_rate': 0.03,
    'num_leaves': 128,        # Больше листьев для сложных зависимостей
    'max_depth': 12,
    'reg_alpha': 2.0,         # L1 регуляризация
    'reg_lambda': 5.0,        # L2 регуляризация
    'colsample_bytree': 0.8,  # Аналог rsm
    'subsample': 0.8,
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
}

# --- 1. ЗАГРУЗКА ---
print("Загрузка данных...")
train_raw = pd.read_csv('data/raw/hackathon_income_train.csv', sep=';', decimal=',', low_memory=False)
test_raw = pd.read_csv('data/raw/hackathon_income_test.csv', sep=';', decimal=',', low_memory=False)
submission = pd.read_csv('data/raw/sample_submission.csv')

# --- 2. УТИЛИТЫ ---
def safe_div(a, b):
    return np.where(b == 0, 0, a / b)

def weighted_mae(y_true, y_pred, weights):
    return (weights * np.abs(y_true - y_pred)).mean()

# --- 3. FEATURE ENGINEERING (SMART) ---
def advanced_fe(df, is_train=False):
    df = df.copy()
    
    # 3.1 Очистка чисел
    # Ищем колонки, которые должны быть числами, но стали object из-за запятых или опечаток
    exclude_cols = ['dt', 'gender', 'adminarea', 'incomeValueCategory', 'id', 'target', 'w']
    num_candidates = df.columns.difference(exclude_cols)
    
    for col in num_candidates:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        # Принудительно в числа
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999.0)

    # 3.2 Даты
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m-%d', errors='coerce')
        df['month'] = df['dt'].dt.month.fillna(-1).astype(int)
        # Глобальный тренд (месяц с начала эры)
        df['month_global'] = (df['dt'].dt.year - 2020) * 12 + df['dt'].dt.month
        df = df.drop(columns=['dt'])
    
    # --- SMART FEATURES (НОВЫЕ) ---
    
    # A. Расхождение зарплат (Mismatch)
    # Зарплата из цифрового профиля vs Зарплата, оцененная банком
    if 'dp_ils_avg_salary_1y' in df.columns and 'salary_6to12m_avg' in df.columns:
        df['salary_mismatch'] = df['dp_ils_avg_salary_1y'] - df['salary_6to12m_avg']
        # Флаг: врет ли клиент? (сильное расхождение)
        df['is_salary_mismatch_big'] = (np.abs(df['salary_mismatch']) > 50000).astype(int)

    # B. Чистый доход (Disposable Income Proxy)
    # ЗП минус обязательства (примерно 5% от лимита карт + кредитов)
    if 'hdb_bki_total_max_limit' in df.columns:
        estimated_payment = df['hdb_bki_total_max_limit'] * 0.05
        df['disposable_income_proxy'] = df['salary_6to12m_avg'] - estimated_payment

    # C. Нагрузка (Debt Burden)
    df['limit_to_income'] = safe_div(df['hdb_bki_total_max_limit'], df['salary_6to12m_avg'])
    
    # D. Активность (Turnover Ratios)
    df['turn_volatility'] = safe_div(df['turn_cur_cr_max_v2'], df['turn_cur_cr_avg_v2'])
    
    # E. Тренды (Короткий vs Длинный период)
    # Растут ли траты? (3 мес vs 12 мес)
    df['trend_turn_cr'] = safe_div(df['avg_cur_cr_turn'], df['turn_cur_cr_avg_act_v2'])

    # 3.3 Категории
    cat_cols = ['gender', 'adminarea', 'incomeValueCategory', 'month']
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Missing")
        
    # Логарифм таргета
    if is_train:
        df['target_log'] = np.log1p(df['target'])

    return df, cat_cols

# Применяем FE
print("Применяем Feature Engineering...")
train_df, cat_features = advanced_fe(train_raw, is_train=True)
test_df, _ = advanced_fe(test_raw, is_train=False)

# Формируем X, y
X = train_df.drop(columns=['target', 'target_log', 'w', 'id'])
y_log = train_df['target_log']
w = train_df['w']
X_test = test_df.drop(columns=['id'])

# Выравниваем колонки
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

# Очистка имен колонок для LightGBM (он не любит спецсимволы JSON)
import re
def clean_names(df):
    new_cols = []
    for c in df.columns:
        c_clean = re.sub(r'[^\w]', '_', c) # Заменяем все не буквы/цифры на _
        new_cols.append(c_clean)
    df.columns = new_cols
    return df

X = clean_names(X)
X_test = clean_names(X_test)
# Обновляем список категорий, так как имена изменились
cat_features = [re.sub(r'[^\w]', '_', c) for c in cat_features]


# --- 4. АВТОМАТИЧЕСКИЙ ОТБОР ПРИЗНАКОВ (Удаление мусора) ---
print("\n--- Этап 1: Экспресс-отбор признаков (удаление importance=0) ---")

# Быстрая модель для оценки
selector_model = CatBoostRegressor(
    iterations=500, 
    learning_rate=0.1, 
    depth=6, 
    loss_function='MAE', 
    verbose=0,
    random_seed=SEED
)

selector_model.fit(X, y_log, cat_features=cat_features, sample_weight=w)

# Получаем важность
fi = selector_model.get_feature_importance(prettified=True)
# Оставляем только те, где важность > 0
useful_feats = fi[fi['Importances'] > 0.0001]['Feature Id'].values.tolist()
dropped_count = X.shape[1] - len(useful_feats)

print(f"Всего признаков: {X.shape[1]}")
print(f"Удалено бесполезных: {dropped_count}")
print(f"Осталось признаков: {len(useful_feats)}")

# Фильтруем датасеты
X = X[useful_feats]
X_test = X_test[useful_feats]
cat_features = [c for c in cat_features if c in useful_feats]

# Подготовка для LightGBM (Label Encoding)
print("Подготовка данных для LightGBM...")
X_lgbm = X.copy()
X_test_lgbm = X_test.copy()

for c in cat_features:
    le = LabelEncoder()
    # Обучаем на объединении train+test, чтобы знать все категории
    vals = list(X[c].astype(str).values) + list(X_test[c].astype(str).values)
    le.fit(vals)
    X_lgbm[c] = le.transform(X[c].astype(str))
    X_test_lgbm[c] = le.transform(X_test[c].astype(str))


# --- 5. ФИНАЛЬНОЕ ОБУЧЕНИЕ (АНСАМБЛЬ) ---
print("\n--- Этап 2: Обучение Ансамбля (CatBoost + LightGBM) ---")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds_cat = np.zeros(len(X))
oof_preds_lgbm = np.zeros(len(X))
test_preds_cat = np.zeros(len(X_test))
test_preds_lgbm = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
    # Данные для фолда
    X_tr, y_tr_log, w_tr = X.iloc[train_idx], y_log.iloc[train_idx], w.iloc[train_idx]
    X_val, y_val_log, w_val = X.iloc[val_idx], y_log.iloc[val_idx], w.iloc[val_idx]
    
    # Данные для LGBM
    X_tr_lgbm, X_val_lgbm = X_lgbm.iloc[train_idx], X_lgbm.iloc[val_idx]

    # --- MODEL 1: CATBOOST ---
    cb = CatBoostRegressor(**CAT_PARAMS)
    cb_pool_tr = Pool(X_tr, y_tr_log, cat_features=cat_features, weight=w_tr)
    cb_pool_val = Pool(X_val, y_val_log, cat_features=cat_features, weight=w_val)
    
    cb.fit(cb_pool_tr, eval_set=cb_pool_val, early_stopping_rounds=200)
    
    # Предикт (сразу экспоненту)
    pred_val_cat = np.expm1(cb.predict(X_val))
    pred_test_cat = np.expm1(cb.predict(X_test))
    
    oof_preds_cat[val_idx] = pred_val_cat
    test_preds_cat += pred_test_cat / N_SPLITS
    
    # --- MODEL 2: LIGHTGBM ---
    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X_tr_lgbm, y_tr_log, sample_weight=w_tr,
             eval_set=[(X_val_lgbm, y_val_log)],
             eval_sample_weight=[w_val],
             callbacks=[]) # Early stopping встроен
             
    pred_val_lgbm = np.expm1(lgbm.predict(X_val_lgbm))
    pred_test_lgbm = np.expm1(lgbm.predict(X_test_lgbm))
    
    oof_preds_lgbm[val_idx] = pred_val_lgbm
    test_preds_lgbm += pred_test_lgbm / N_SPLITS
    
    # --- BLEND SCORE ---
    # 60% CatBoost, 40% LightGBM
    blend_val = 0.6 * pred_val_cat + 0.4 * pred_val_lgbm
    y_true_orig = np.expm1(y_val_log)
    
    score = weighted_mae(y_true_orig, blend_val, w_val)
    scores.append(score)
    
    print(f"\nFold {fold+1} WMAE: {score:.2f} (Cat: {weighted_mae(y_true_orig, pred_val_cat, w_val):.2f} / LGBM: {weighted_mae(y_true_orig, pred_val_lgbm, w_val):.2f})")
    
    # Чистим память
    del cb, lgbm, cb_pool_tr, cb_pool_val
    gc.collect()

# --- 6. РЕЗУЛЬТАТЫ И СОХРАНЕНИЕ ---
avg_score = np.mean(scores)
print(f"\n==========================================")
print(f"Средний CV WMAE (Ensemble): {avg_score:.2f}")
print(f"==========================================")

# Финальный блендинг для теста
final_test_pred = 0.6 * test_preds_cat + 0.4 * test_preds_lgbm
# Защита от отрицательных значений
final_test_pred = np.maximum(0, final_test_pred)

submission['target'] = final_test_pred
filename = f'submission_ensemble_autoselect_cv{avg_score:.0f}.csv'
submission.to_csv(f'data/processed/{filename}', index=False)
print(f"Файл сохранен как: {filename}")