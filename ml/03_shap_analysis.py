import pandas as pd
import numpy as np
import shap
import pickle
import os
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Should be project root
DATA_DIR = os.path.join(BASE_DIR, 'ml', 'data')

def load_training_info():
    """Загрузка информации об обучении"""
    try:
        with open(os.path.join(DATA_DIR, 'processed', 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # Загружаем только необходимые колонки для экономии памяти
        test = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'test_processed.csv'), usecols=['id'] + model_info['features'])
        X_test = test.drop(columns=['id'])
        
        # Берем только общие фичи
        X_test = X_test[model_info['features']]
        
        return model_info, X_test, test
    
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return None, None, None

def calculate_shap():
    print("=== 3. SHAP АНАЛИЗ ===")
    
    try:
        # Загрузка данных
        model_info, X_test, test_df = load_training_info()
        if model_info is None:
            return
        
        print(f"Размер данных: {X_test.shape}")
        
        # Ограничиваем количество строк для SHAP анализа
        SAMPLE_SIZE = 1000  # Можно увеличить если хватит памяти
        if len(X_test) > SAMPLE_SIZE:
            print(f"Берем выборку из {SAMPLE_SIZE} строк для SHAP анализа")
            sample_indices = np.random.choice(len(X_test), SAMPLE_SIZE, replace=False)
            X_sample = X_test.iloc[sample_indices].copy()
            test_sample = test_df.iloc[sample_indices].copy()
        else:
            X_sample = X_test.copy()
            test_sample = test_df.copy()
        
        # Загружаем первую модель для SHAP
        print("Загрузка модели для SHAP анализа...")
        
        # Adjust model path if needed
        model_path = model_info['model_paths'][0]
        if not os.path.exists(model_path) and model_path.startswith('data/'):
             model_path = os.path.join(BASE_DIR, 'ml', model_path)
        elif not os.path.exists(model_path):
             # Try to find in DATA_DIR
             model_name = os.path.basename(model_path)
             model_path = os.path.join(DATA_DIR, 'models', 'catboost_models', model_name)

        model = CatBoostRegressor()
        model.load_model(model_path)
        
        # Используем приближенный SHAP для экономии памяти
        print("Вычисление приближенных SHAP значений...")
        explainer = shap.TreeExplainer(model)
        
        # Вычисляем SHAP значения для выборки
        shap_values = explainer.shap_values(X_sample)
        
        # Создаем DataFrame с SHAP значениями
        shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
        shap_df['id'] = test_sample['id'].values
        
        # Загружаем предсказания только для выборки
        submission = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'submission_wmae.csv'))
        # We need to match by ID because indices might differ if submission was reordered or if we sampled
        # But here test_sample came from test_processed, submission came from submission_wmae
        # Assuming they are aligned or we merge by ID. Merging is safer.
        submission_sample = submission[submission['id'].isin(shap_df['id'])].set_index('id')
        # Map targets back to shap_df order
        shap_df['predicted_income'] = shap_df['id'].map(submission_sample['target'])
        
        # Сохраняем SHAP значения
        shap_df.to_csv(os.path.join(DATA_DIR, 'processed', 'shap_values.csv'), index=False) # Using main file name consistent with other scripts
        
        # Сохраняем средние абсолютные SHAP значения (важность признаков)
        shap_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0),
            'mean_shap': shap_values.mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        shap_importance.to_csv(os.path.join(DATA_DIR, 'processed', 'shap_importance.csv'), index=False)
        
        print("✅ SHAP анализ завершен!")
        print(f"Сохранено SHAP значений для {len(shap_df)} клиентов")
        print(f"Топ-10 самых важных признаков:")
        print(shap_importance.head(10))
        
    except MemoryError:
        print("❌ Недостаточно памяти для SHAP анализа")
        print("Попробуйте уменьшить SAMPLE_SIZE или использовать более мощную машину")
    
    except Exception as e:
        print(f"❌ Ошибка при выполнении SHAP анализа: {e}")
        import traceback
        traceback.print_exc()

def create_frontend_shap_data():
    """Создает оптимизированный файл для фронтенда"""
    try:
        print("Создание файла для фронтенда...")
        
        # Загружаем SHAP данные
        shap_df = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'shap_values.csv'))
        
        # Создаем компактную версию только с топ-признаками
        TOP_FEATURES_FRONTEND = 15  # Для фронтенда показываем только топ-15
        
        # Вычисляем среднюю абсолютную важность признаков
        feature_importance = {}
        for col in shap_df.columns:
            if col not in ['id', 'predicted_income']:
                feature_importance[col] = np.abs(shap_df[col]).mean()
        
        # Сортируем по важности
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:TOP_FEATURES_FRONTEND]
        top_feature_names = [feat[0] for feat in top_features]
        
        # Создаем компактный DataFrame только с топ-признаками
        frontend_df = shap_df[['id', 'predicted_income'] + top_feature_names].copy()
        
        # Добавляем человеко-читаемые названия признаков (опционально)
        feature_mapping = {
            # Добавь маппинг русских названий для ключевых признаков
            # 'some_english_feature': 'Человеко-читаемое название'
        }
        
        frontend_df.rename(columns=feature_mapping, inplace=True)
        frontend_df.to_csv(os.path.join(DATA_DIR, 'processed', 'shap_values_frontend.csv'), index=False)
        
        print(f"✅ Файл для фронтенда создан!")
        print(f"   Сохранено {len(frontend_df)} клиентов")
        print(f"   Используется {len(top_feature_names)} самых важных признаков")
        print(f"   Топ-5 признаков: {top_feature_names[:5]}")
        
    except Exception as e:
        print(f"❌ Ошибка при создании файла для фронтенда: {e}")


if __name__ == "__main__":
    calculate_shap()
    create_frontend_shap_data()