import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool

# --- 1. Загрузка и подготовка (упрощенная) ---
print("Загрузка данных для отбора признаков...")
train_raw = pd.read_csv('data/raw/hackathon_income_train.csv', sep=';', decimal=',', low_memory=False)

# Быстрая предобработка (тот же пайплайн, что и раньше)
def fast_preprocessing(df):
    # Числа
    num_cols = df.columns.difference(['dt', 'gender', 'adminarea', 'incomeValueCategory', 'target', 'w', 'id']).tolist()
    for col in num_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999.0)
    
    # Категории
    cat_cols = ['gender', 'adminarea', 'incomeValueCategory']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Missing")
            
    return df, cat_cols

df, cat_features = fast_preprocessing(train_raw)

# Готовим X и y
X = df.drop(columns=['target', 'w', 'id', 'dt'], errors='ignore')
y = np.log1p(df['target']) # Учимся на логарифме
w = df['w']

# Убедимся, что категории есть в X
cat_features = [c for c in cat_features if c in X.columns]

print(f"Всего признаков на входе: {X.shape[1]}")

# --- 2. Обучение быстрой модели для оценки важности ---
# Нам не нужен кросс-валидация и супер-точность, нам нужен Feature Importance
model = CatBoostRegressor(
    iterations=1000,          # Хватит, чтобы понять суть
    learning_rate=0.1,
    depth=6,
    loss_function='MAE',
    cat_features=cat_features,
    random_seed=42,
    verbose=100
)

print("Запуск анализа важности...")
model.fit(X, y, sample_weight=w)

# --- 3. Получение и Фильтрация ---
feature_importance = model.get_feature_importance(prettified=True)

# Смотрим на "хвост"
print("\n--- Топ 10 Признаков ---")
print(feature_importance.head(10))

print("\n--- Худшие 10 Признаков ---")
print(feature_importance.tail(10))

# Фильтр: берем только те, где важность > 0 (или порог типа 0.01)
# Часто помогает убрать те, что имеют важность 0.0
useful_features = feature_importance[feature_importance['Importances'] > 0.0]['Feature Id'].tolist()
useless_features = feature_importance[feature_importance['Importances'] == 0.0]['Feature Id'].tolist()

print(f"\nВсего признаков: {len(feature_importance)}")
print(f"Полезных признаков (>0): {len(useful_features)}")
print(f"Бесполезных признаков (=0): {len(useless_features)}")

# Сохраняем список полезных признаков
import pickle
with open('data/processed/useful_features_list.pkl', 'wb') as f:
    pickle.dump(useful_features, f)

print("Список полезных признаков сохранен в 'data/processed/useful_features_list.pkl'")

# --- 4. Визуализация (опционально) ---
plt.figure(figsize=(12, 6))
sns.barplot(x="Importances", y="Feature Id", data=feature_importance.head(20))
plt.title('Top 20 Feature Importances')
plt.show()