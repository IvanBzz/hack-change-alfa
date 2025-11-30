// Global variables
let clientsData = [];
let shapData = [];
let recommendationsData = [];
let shapChart = null;

// DOM elements
const clientIdInput = document.getElementById('clientIdInput');
const searchBtn = document.getElementById('searchBtn');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const clientIdDisplay = document.getElementById('clientIdDisplay');
const incomeValue = document.getElementById('incomeValue');
const recommendationsList = document.getElementById('recommendationsList');

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Сервис прогнозирования доходов инициализирован');
    loadData();
    setupEventListeners();
});

function setupEventListeners() {
    searchBtn.addEventListener('click', handleSearch);
    
    clientIdInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // Real-time input validation - только цифры
    clientIdInput.addEventListener('input', function() {
        this.value = this.value.replace(/[^0-9]/g, '');
    });
}

async function loadData() {
    try {
        console.log('Загрузка данных...');
        
        // Загружаем все три файла параллельно
        const [submissionData, shapDataResponse, recommendationsDataResponse] = await Promise.all([
            loadCSV('data/processed/submission_wmae.csv'),
            loadCSV('data/processed/shap_values.csv'),
            loadCSV('data/processed/client_recommendations.csv')
        ]);

        clientsData = submissionData || [];
        shapData = shapDataResponse || [];
        recommendationsData = recommendationsDataResponse || [];

        console.log('✅ Данные успешно загружены:');
        console.log(`- Клиентов: ${clientsData.length}`);
        console.log(`- SHAP данных: ${shapData.length}`);
        console.log(`- Рекомендаций: ${recommendationsData.length}`);
        
        // Выводим примеры данных для отладки
        if (clientsData.length > 0) {
            console.log('Пример клиента:', clientsData[0]);
        }
        if (shapData.length > 0) {
            console.log('Пример SHAP данных:', Object.keys(shapData[0]).slice(0, 5));
        }
        if (recommendationsData.length > 0) {
            console.log('Пример рекомендаций:', recommendationsData[0]);
        }
        
    } catch (error) {
        console.error('❌ Ошибка загрузки данных:', error);
        // Используем тестовые данные если загрузка не удалась
        loadTestData();
    }
}

function loadCSV(filePath) {
    return new Promise((resolve, reject) => {
        Papa.parse(filePath, {
            download: true,
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                if (results.errors.length > 0) {
                    console.warn('Предупреждения при парсинге CSV:', results.errors);
                }
                console.log(`Загружено ${results.data.length} строк из ${filePath}`);
                if (results.meta.fields) {
                    console.log('Колонки:', results.meta.fields.slice(0, 5), '...');
                }
                resolve(results.data);
            },
            error: function(error) {
                console.error(`Ошибка загрузки файла ${filePath}:`, error);
                resolve([]); // Возвращаем пустой массив вместо ошибки
            }
        });
    });
}

function handleSearch() {
    const clientId = clientIdInput.value.trim();
    
    console.log('🔄 Поиск клиента с ID:', clientId);
    console.log('Доступные данные:', {
        clients: clientsData.length,
        shap: shapData.length,
        recommendations: recommendationsData.length
    });
    
    if (!clientId) {
        showError('Пожалуйста, введите ID клиента');
        return;
    }
    
    searchClient(clientId);
}

function searchClient(clientId) {
    showLoading(true);
    
    // Имитация загрузки для лучшего UX
    setTimeout(() => {
        const client = findClientData(clientId);
        
        if (client) {
            console.log(`✅ Клиент найден:`, client);
            displayClientData(client);
        } else {
            console.log(`❌ Клиент с ID ${clientId} не найден`);
            showClientNotFound();
        }
        
        showLoading(false);
    }, 800);
}

function findClientData(clientId) {
    console.log('Поиск клиента ID:', clientId);
    
    // Для submission_wmae.csv - ищем по id
    const clientSubmission = clientsData.find(c => c.id == clientId);
    console.log('Найден в submission:', clientSubmission);
    
    // Для shap_values.csv - ID как число или строка
    const clientShap = shapData.find(s => s.id == clientId);
    console.log('Найден в SHAP:', clientShap ? 'да' : 'нет');
    
    // Для client_recommendations.csv - ID как число или строка
    const clientRecommendations = recommendationsData.find(r => r.id == clientId);
    console.log('Найден в рекомендациях:', clientRecommendations ? 'да' : 'нет');
    
    if (clientSubmission) {
        return {
            id: clientId,
            submission: clientSubmission,
            shap: clientShap,
            recommendations: clientRecommendations
        };
    }
    
    return null;
}

function displayClientData(client) {
    hideError();
    
    // Обновляем информацию о клиенте
    clientIdDisplay.textContent = client.id;
    
    // Обновляем прогноз дохода
    const income = client.submission.target || client.submission.predicted_income;
    incomeValue.textContent = formatCurrency(income);
    
    // Создаем SHAP график если есть данные
    if (client.shap) {
        createShapChart(client.shap);
    } else {
        console.warn('SHAP данные не найдены для клиента');
        showShapPlaceholder();
    }
    
    // Показываем рекомендации
    displayRecommendations(client.recommendations);
    
    // Показываем секцию с результатами
    resultsSection.classList.remove('hidden');
    
    // Плавно скроллим к результатам
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createShapChart(shapData) {
    const ctx = document.getElementById('shapChart').getContext('2d');
    
    // Удаляем предыдущий график если есть
    if (shapChart) {
        shapChart.destroy();
    }
    
    // Подготавливаем данные для графика
    const features = [];
    const impacts = [];
    
    // Собираем все фичи кроме служебных полей
    for (const [key, value] of Object.entries(shapData)) {
        // Исключаем служебные поля и проверяем что значение число
        if (key !== 'id' && key !== 'predicted_income' && 
            typeof value === 'number' && !isNaN(value) && value !== 0) {
            features.push(key);
            impacts.push(value);
        }
    }
    
    console.log(`Найдено ${features.length} фичей с SHAP значениями`);
    
    if (features.length === 0) {
        showShapPlaceholder();
        return;
    }
    
    // Сортируем по абсолютному влиянию и берем топ-10
    const featureImpacts = features.map((feature, index) => ({
        feature: formatFeatureName(feature),
        impact: impacts[index],
        absoluteImpact: Math.abs(impacts[index])
    })).sort((a, b) => b.absoluteImpact - a.absoluteImpact).slice(0, 10);
    
    const sortedFeatures = featureImpacts.map(item => item.feature);
    const sortedImpacts = featureImpacts.map(item => item.impact);
    
    console.log('Топ фичи для графика:', sortedFeatures);
    console.log('SHAP значения:', sortedImpacts);
    
    // Цвета в зависимости от направления влияния
    const backgroundColors = sortedImpacts.map(impact => 
        impact >= 0 ? 'rgba(169, 239, 1, 0.8)' : 'rgba(239, 49, 36, 0.8)'
    );
    
    const borderColors = sortedImpacts.map(impact => 
        impact >= 0 ? 'rgb(169, 239, 1)' : 'rgb(239, 49, 36)'
    );
    
    shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedFeatures,
            datasets: [{
                label: 'Влияние на доход (руб.)',
                data: sortedImpacts,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const impact = context.raw;
                            const direction = impact >= 0 ? '📈 увеличивает' : '📉 уменьшает';
                            return `${direction} доход на ${formatCurrency(Math.abs(impact))}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Влияние на прогноз дохода (руб.)',
                        font: {
                            size: 14,
                            weight: '600'
                        },
                        color: '#666'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                }
            },
            animation: {
                duration: 1200,
                easing: 'easeOutQuart'
            }
        }
    });
}

function showShapPlaceholder() {
    const chartContainer = document.querySelector('.chart-container');
    chartContainer.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; height: 100%; flex-direction: column; gap: 1rem;">
            <div style="font-size: 4rem;">📊</div>
            <h3 style="color: var(--text-secondary); text-align: center;">
                SHAP данные временно недоступны<br>
                <small style="font-weight: normal;">График будет отображен при наличии данных</small>
            </h3>
        </div>
    `;
}

function displayRecommendations(recommendations) {
    recommendationsList.innerHTML = '';
    
    if (recommendations && recommendations.recommendations) {
        const recs = recommendations.recommendations.split('|');
        
        recs.forEach((rec, index) => {
            if (rec.trim()) {
                const recommendationItem = document.createElement('div');
                recommendationItem.className = 'recommendation-item';
                recommendationItem.innerHTML = `
                    <h4>💰 Предложение ${index + 1}</h4>
                    <p>${rec.trim()}</p>
                `;
                recommendationsList.appendChild(recommendationItem);
            }
        });
        
        if (recommendationsList.children.length === 0) {
            showRecommendationsPlaceholder();
        }
    } else {
        showRecommendationsPlaceholder();
    }
}

function showRecommendationsPlaceholder() {
    recommendationsList.innerHTML = `
        <div class="recommendation-item">
            <h4>💡 Персонализированные рекомендации</h4>
            <p>На основе прогнозируемого дохода клиента формируются индивидуальные предложения по финансовым продуктам Альфа-Банка</p>
        </div>
    `;
}

function formatCurrency(amount) {
    if (isNaN(amount)) return '0 ₽';
    
    return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'RUB',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function formatFeatureName(featureName) {
    // Расширенный маппинг английских названий на русские
    const featureMap = {
        'age': 'Возраст',
        'work_experience': 'Стаж работы',
        'has_mortgage': 'Наличие ипотеки',
        'incomelevel': 'Уровень дохода',
        'creditscore': 'Кредитный рейтинг',
        'employmenttype': 'Тип занятости',
        'educationlevel': 'Образование',
        'familystatus': 'Семейное положение',
        'city': 'Город проживания',
        'previousloans': 'История кредитов',
        'accountbalance': 'Баланс счета',
        'transactionfrequency': 'Активность операций',
        'salary': 'Зарплата',
        'turn_cur_cr_avg_act_v2': 'Оборот по кредитам',
        'hdb_bki_total_max_limit': 'Макс. лимит кредитов',
        'dp_ils_paymentssum_avg_12m': 'Средние платежи за 12 мес.',
        'month': 'Месяц',
        'year': 'Год',
        'day_of_week': 'День недели',
        'dt': 'Дата'
    };
    
    // Ищем совпадение (регистронезависимо)
    const lowerFeature = featureName.toLowerCase();
    for (const [eng, rus] of Object.entries(featureMap)) {
        if (lowerFeature.includes(eng)) {
            return rus;
        }
    }
    
    // Если не нашли маппинг, форматируем оригинальное название
    return featureName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function showLoading(show) {
    const btnText = searchBtn.querySelector('.btn-text');
    const btnLoader = searchBtn.querySelector('.btn-loader');
    
    if (show) {
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        searchBtn.disabled = true;
        searchBtn.style.opacity = '0.8';
    } else {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        searchBtn.disabled = false;
        searchBtn.style.opacity = '1';
    }
}

function showClientNotFound() {
    showError('Клиент с указанным ID не зарегистрирован в системе. Пожалуйста, проверьте правильность введенного ID.');
    clientIdInput.focus();
}

function showError(message) {
    errorSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    
    const errorText = errorSection.querySelector('p');
    if (errorText) {
        errorText.textContent = message;
    }
    
    // Плавно скроллим к ошибке
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() {
    errorSection.classList.add('hidden');
    clientIdInput.focus();
}

// Тестовые данные для отладки
function loadTestData() {
    console.log('Загрузка тестовых данных...');
    
    clientsData = [
        {id: "0", target: 59504.31},
        {id: "1", target: 53657.56},
        {id: "2", target: 72345.89},
        {id: "3", target: 48912.45}
    ];
    
    shapData = [
        {
            id: "0", 
            age: 1500, 
            salary_6to12m_avg: -3456, 
            turn_cur_cr_avg_act_v2: 16283,
            work_experience: 8500,
            education_level: 2800,
            has_mortgage: -5200
        },
        {
            id: "1", 
            age: -4297, 
            salary_6to12m_avg: -4231, 
            turn_cur_cr_avg_act_v2: -3734,
            work_experience: 12500,
            education_level: 3200,
            has_mortgage: -2800
        }
    ];
    
    recommendationsData = [
        {
            id: "0", 
            recommendations: "Кредитная карта 'Премиум' с лимитом 300 000 руб. | Страхование жизни | Инвестиционный брокерский счет"
        },
        {
            id: "1", 
            recommendations: "Кредитная карта 'Старт' с лимитом 100 000 руб. | Накопительный счет с повышенной ставкой | Бесплатное обслуживание карты"
        }
    ];
    
    console.log('✅ Тестовые данные загружены');
}

// Глобальные функции для кнопок
window.hideError = hideError;