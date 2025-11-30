import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# --- 1. Функция метрики (из бейзлайна организаторов) ---
def custom_wmae_metric(y_true, y_pred, weights):
    # Организаторы используют .mean(), что делит на количество строк (N)
    return (weights * np.abs(y_true - y_pred)).mean()

# --- 2. Загрузка и Предобработка ---
print("Загрузка данных...")
train = pd.read_csv('data/raw/hackathon_income_train.csv', sep=';', decimal=',', low_memory=False)
test = pd.read_csv('data/raw/hackathon_income_test.csv', sep=';', decimal=',', low_memory=False)
submission = pd.read_csv('data/raw/sample_submission.csv')

def preprocess_data(df):
    # Удаляем дату, извлекаем полезное, если нужно (тут упростим, чтобы не шуметь)
    if 'dt' in df.columns:
        df = df.drop(columns=['dt'])
    
    # Работа с категориями:
    # 1. Находим колонки типа Object
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # 2. Приводим всё к строке, чтобы убрать mixed types (числа+строки)
    # Это лечит ошибку "bad object for id"
    for col in cat_features:
        df[col] = df[col].astype(str).fillna("Missing")
        
    return df, cat_features

train, cat_cols = preprocess_data(train)
test, _ = preprocess_data(test)

# Определяем X, y, w
X = train.drop(columns=['target', 'w', 'id'])
y = train['target'] # Работаем с чистым таргетом, без логарифма!
w = train['w']      # Веса

X_test = test.drop(columns=['id'])
# Оставляем только общие колонки
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]
cat_features = [c for c in cat_cols if c in X.columns]

print(f"Признаков для обучения: {X.shape[1]}")

# --- 3. OPTUNA: Подбор гиперпараметров ---
def objective(trial):
    # Сплит для валидации внутри оптюны (один фолд для скорости)
    train_x, valid_x = X.iloc[:-15000], X.iloc[-15000:]
    train_y, valid_y = y.iloc[:-15000], y.iloc[-15000:]
    train_w, valid_w = w.iloc[:-15000], w.iloc[-15000:]
    
    # Сетка параметров для перебора
    params = {
        'iterations': 1500, # Не слишком много для теста
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'CPU',
        'random_seed': 42,
        'verbose': False
    }
    
    train_pool = Pool(train_x, train_y, weight=train_w, cat_features=cat_features)
    valid_pool = Pool(valid_x, valid_y, weight=valid_w, cat_features=cat_features)
    
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100)
    
    preds = model.predict(valid_x)
    # Считаем метрику организаторов
    score = custom_wmae_metric(valid_y, preds, valid_w)
    
    return score

print("\n--- Запуск Optuna (это может занять время) ---")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20) # 20 прогонов (поставь больше, если есть время, например 50)

print(f"\nЛучшие параметры: {study.best_params}")
print(f"Лучшая метрика (Local WMAE): {study.best_value}")

# --- 4. Финальное обучение с лучшими параметрами ---
print("\n--- Финальное обучение ---")
best_params = study.best_params
best_params['iterations'] = 3000 # Для финала ставим побольше
best_params['loss_function'] = 'MAE'
best_params['eval_metric'] = 'MAE'
best_params['random_seed'] = 42
best_params['verbose'] = 500

# K-Fold обучение для надежности
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
oof_score = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train, w_train = X.iloc[train_idx], y.iloc[train_idx], w.iloc[train_idx]
    X_val, y_val, w_val = X.iloc[val_idx], y.iloc[val_idx], w.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, weight=w_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, weight=w_val, cat_features=cat_features)
    
    model = CatBoostRegressor(**best_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=200)
    
    # Валидация
    val_pred = model.predict(X_val)
    fold_score = custom_wmae_metric(y_val, val_pred, w_val)
    oof_score.append(fold_score)
    print(f"Fold {fold+1} WMAE: {fold_score}")
    
    # Предикт на тест
    test_preds += model.predict(X_test) / 5

print(f"\nСредняя метрика на кросс-валидации: {np.mean(oof_score)}")

# --- 5. Сохранение ---
submission['target'] = test_preds
submission.to_csv('data/processed/submission_optuna_v6.csv', index=False)
print("Файл submission_optuna_v3.csv готов к отправке!")