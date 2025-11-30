import pandas as pd
import numpy as np

def generate_recommendations(predicted_income, top_features, shap_values_row):
    """Генерация рекомендаций на основе дохода и важных фичей"""
    recommendations = []
    
    # Правила на основе дохода
    if predicted_income < 50000:
        recommendations.append("Кредитная карта 'Старт' с лимитом 100 000 руб.")
        recommendations.append("Накопительный счет с повышенной ставкой")
        recommendations.append("Бесплатное обслуживание карты")
    elif predicted_income < 100000:
        recommendations.append("Кредитная карта 'Премиум' с лимитом 300 000 руб.")
        recommendations.append("Потребительский кредит до 1 000 000 руб.")
        recommendations.append("Инвестиционный брокерский счет")
        recommendations.append("Страхование жизни")
    else:
        recommendations.append("Кредитная карта 'Премиум+' с лимитом 700 000 руб.")
        recommendations.append("Ипотека с пониженной ставкой")
        recommendations.append("Премиальная дебетовая карта с кэшбэком")
        recommendations.append("Индивидуальное инвестиционное предложение")
        recommendations.append("Private banking обслуживание")
    
    # Дополнительные рекомендации на основе важных фичей
    for feature, impact in top_features[:3]:  # Берем 3 самых влиятельных фактора
        feature_lower = str(feature).lower()
        
        if any(word in feature_lower for word in ['возраст', 'age']):
            if impact > 0:
                recommendations.append("Пенсионная программа (досрочное накопление)")
            else:
                recommendations.append("Молодежная кредитная карта")
                
        elif any(word in feature_lower for word in ['ипотек', 'mortgage', 'недвиж']):
            if impact < 0:
                recommendations.append("Рефинансирование ипотеки")
            else:
                recommendations.append("Страхование недвижимости")
                
        elif any(word in feature_lower for word in ['стаж', 'experience', 'работ']):
            if impact > 0:
                recommendations.append("Кредит на развитие бизнеса")
            else:
                recommendations.append("Программа поддержки начинающих предпринимателей")
                
        elif any(word in feature_lower for word in ['семь', 'family', 'дет']):
            if impact > 0:
                recommendations.append("Семейная ипотека")
                recommendations.append("Детский накопительный счет")
    
    return list(set(recommendations))  # Убираем дубли

def create_fallback_recommendations():
    """Создает рекомендации на основе только предсказанного дохода"""
    print("Создание рекомендаций на основе дохода...")
    
    try:
        submission = pd.read_csv('data/processed/submission_wmae.csv')
        recommendations_list = []
        
        for _, row in submission.iterrows():
            recommendations = generate_recommendations(row['target'], [], [])
            recommendations_list.append({
                'id': row['id'],
                'predicted_income': row['target'],
                'top_features': 'Анализ факторов временно недоступен',
                'recommendations': ' | '.join(recommendations)
            })
        
        pd.DataFrame(recommendations_list).to_csv('data/processed/client_recommendations.csv', index=False)
        print("✅ Рекомендации на основе дохода созданы!")
        
    except Exception as e:
        print(f"❌ Ошибка при создании заглушки: {e}")

def main():
    print("=== 4. ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ ===")
    
    try:
        # Пробуем сначала основной файл SHAP
        print("Попытка загрузить SHAP данные...")
        shap_df = pd.read_csv('data/processed/shap_values.csv')
        
    except FileNotFoundError:
        try:
            # Пробуем backup файл
            print("Файл shap_values.csv не найден, пробуем shap_values_sample.csv...")
            shap_df = pd.read_csv('data/processed/shap_values_sample.csv')
            
        except FileNotFoundError:
            print("❌ SHAP файлы не найдены. Создаем рекомендации только на основе дохода.")
            create_fallback_recommendations()
            return
    
    # Если SHAP данные загружены успешно
    recommendations_list = []
    
    print("Генерация рекомендаций...")
    for i, row in shap_df.iterrows():
        client_id = row['id']
        predicted_income = row['predicted_income']
        
        # Получаем топ-5 самых влиятельных фичей
        shap_values = []
        for feature in row.index:
            if feature not in ['id', 'predicted_income'] and pd.notna(row[feature]):
                shap_values.append((feature, row[feature]))
        
        # Сортируем по абсолютному влиянию
        shap_values.sort(key=lambda x: abs(x[1]), reverse=True)
        top_5_features = shap_values[:5]
        
        # Генерируем рекомендации
        recommendations = generate_recommendations(predicted_income, shap_values, row)
        
        recommendations_list.append({
            'id': client_id,
            'predicted_income': predicted_income,
            'top_features': '; '.join([f"{feat}: {impact:.0f} руб." for feat, impact in top_5_features]),
            'recommendations': ' | '.join(recommendations)
        })
        
        if (i + 1) % 100 == 0:
            print(f"Обработано {i + 1} клиентов...")
    
    # Сохраняем рекомендации
    recommendations_df = pd.DataFrame(recommendations_list)
    recommendations_df.to_csv('data/processed/client_recommendations.csv', index=False)
    
    print("✅ Генерация рекомендаций завершена!")
    print(f"Создано рекомендаций для {len(recommendations_df)} клиентов")
    
    # Покажем пример
    print("\nПример рекомендаций для первых 3 клиентов:")
    print(recommendations_df.head(3)[['id', 'predicted_income', 'recommendations']])

if __name__ == "__main__":
    main()