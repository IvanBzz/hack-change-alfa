import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# --- 1. Загрузка данных ---
train = pd.read_csv('data/raw/hackathon_income_train.csv', sep=';', decimal=',',low_memory=False)
test = pd.read_csv('data/raw/hackathon_income_test.csv', sep=';', decimal=',',low_memory=False)
submission = pd.read_csv('data/raw/sample_submission.csv', sep=',')

print(f"Размер трейна: {train.shape}")
print(f"Размер теста: {test.shape}")

def preprocess_data(df):
    # --- Работа с датами ---
    if 'dt' in df.columns:
        # Пытаемся распарсить дату. Если ошибка, ставим NaT
        df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
        df['month'] = df['dt'].dt.month
        df['year'] = df['dt'].dt.year
        df['day_of_week'] = df['dt'].dt.dayofweek
        df = df.drop(columns=['dt'])
    
    # --- Заполнение пропусков и приведение типов ---
    
    # 1. Сначала выделим "настоящие" категориальные признаки (тип object)
    # Важно: делаем копию списка, чтобы не зависеть от изменений df на лету
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # 2. Проходим по каждому категориальному признаку и приводим к строке
    for col in cat_features:
        # Заполняем пропуски строкой "Missing"
        df[col] = df[col].fillna("Missing")
        # ПРИНУДИТЕЛЬНО приводим к строке. Это решает проблему с "55.0"
        df[col] = df[col].astype(str)
        
    # Числовые признаки (на всякий случай заполним пропуски, CatBoost это умеет, но лучше явно)
    # num_features = df.select_dtypes(exclude=['object']).columns.tolist()
    # df[num_features] = df[num_features].fillna(0) # Или средним, по желанию

    return df, cat_features

# Объединяем для единообразия, но можно и раздельно
train, cat_cols = preprocess_data(train)
test, _ = preprocess_data(test)


# --- 4. Подготовка к обучению ---
X = train.drop(columns=['target', 'w', 'id']) # Удаляем лишнее
y = train['target']
w = train['w'] # Наши веса для WMAE

X_test = test.drop(columns=['id'])
# Убедимся, что колонки совпадают
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

# Обновляем список категориальных признаков (только те, что остались)
cat_features = [c for c in cat_cols if c in X.columns]

# --- 5. Обучение модели (Cross-Validation) ---
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")
    
    X_train, y_train, w_train = X.iloc[train_idx], y.iloc[train_idx], w.iloc[train_idx]
    X_val, y_val, w_val = X.iloc[val_idx], y.iloc[val_idx], w.iloc[val_idx]
    
    # Создаем Pool для CatBoost с указанием весов!
    train_pool = Pool(data=X_train, label=y_train, weight=w_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, weight=w_val, cat_features=cat_features)
    
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE', # Оптимизируем MAE с весами = WMAE
        eval_metric='MAE',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=100,
        task_type="CPU" # Если есть GPU, поменяй на "GPU"
    )
    
    model.fit(train_pool, eval_set=val_pool)
    
    # Предсказание
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    
    # Предсказание на тест (суммируем и потом усредним)
    test_preds += model.predict(X_test) / n_folds
    models.append(model)

    # Расчет метрики на фолде вручную для проверки
    # WMAE = sum(w * |y - y_pred|) / sum(w)
    w_val_sum = np.sum(w_val)
    if w_val_sum > 0:
        fold_wmae = np.sum(w_val * np.abs(y_val - val_pred)) / w_val_sum
        print(f"Fold {fold + 1} WMAE: {fold_wmae:.4f}")
    else:
        print(f"⚠️ ВНИМАНИЕ: Fold {fold + 1} - сумма весов равна нулю!")

# --- 6. Итоговая метрика и создание сабмита ---
w_sum = np.sum(w)
if w_sum > 0:
    total_wmae = np.sum(w * np.abs(y - oof_preds)) / w_sum
    print(f"\n✅ Итоговый OOF WMAE: {total_wmae:.4f}")
else:
    print("\n⚠️ ВНИМАНИЕ: Сумма весов равна нулю! Невозможно рассчитать WMAE.")

# Формируем файл посылки
submission['target'] = test_preds
submission.to_csv('data/processed/submission_wmae.csv', index=False)
print("Файл submission_wmae.csv успешно сохранен.")