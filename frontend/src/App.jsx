import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { Search, Loader2, Calculator, User, RefreshCcw, Briefcase, Wallet, CreditCard, ChevronDown, ChevronUp } from 'lucide-react';
import ParamInput from './components/ParamInput';
import './index.css';ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// --- Helper Functions ---

const formatCurrency = (amount) => {
  if (isNaN(amount)) return '0 ₽';
  return new Intl.NumberFormat('ru-RU', {
    style: 'currency',
    currency: 'RUB',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
};

const formatFeatureName = (featureName) => {
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

  const lowerFeature = featureName.toLowerCase();
  for (const [eng, rus] of Object.entries(featureMap)) {
    if (lowerFeature.includes(eng)) {
      return rus;
    }
  }

  return featureName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// --- Components ---

const ShapChart = ({ shapData }) => {
  if (!shapData) return null;

  const features = [];
  const impacts = [];

  for (const [key, value] of Object.entries(shapData)) {
    if (key !== 'id' && key !== 'predicted_income' && typeof value === 'number' && !isNaN(value) && value !== 0) {
      features.push(key);
      impacts.push(value);
    }
  }

  if (features.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-gray-500">
        <div className="text-6xl">📊</div>
        <h3 className="text-center font-medium">SHAP данные временно недоступны</h3>
      </div>
    );
  }

  const featureImpacts = features.map((feature, index) => ({
    feature: formatFeatureName(feature),
    impact: impacts[index],
    absoluteImpact: Math.abs(impacts[index])
  })).sort((a, b) => b.absoluteImpact - a.absoluteImpact).slice(0, 10);

  const sortedFeatures = featureImpacts.map(item => item.feature);
  const sortedImpacts = featureImpacts.map(item => item.impact);

  const backgroundColors = sortedImpacts.map(impact => impact >= 0 ? 'rgba(169, 239, 1, 0.8)' : 'rgba(239, 49, 36, 0.8)');
  const borderColors = sortedImpacts.map(impact => impact >= 0 ? 'rgb(169, 239, 1)' : 'rgb(239, 49, 36)');

  const data = {
    labels: sortedFeatures,
    datasets: [{
      label: 'Влияние на доход (руб.)',
      data: sortedImpacts,
      backgroundColor: backgroundColors,
      borderColor: borderColors,
      borderWidth: 2,
      borderRadius: 6,
    }]
  };

  const options = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => {
            const impact = context.raw;
            const direction = impact >= 0 ? '📈 увеличивает' : '📉 уменьшает';
            return `${direction} оценку на ${Math.abs(impact).toFixed(4)}`;
          }
        }
      }
    },
    scales: {
      x: {
        title: { display: true, text: 'Сила влияния фактора (Log-шкала)', color: '#94A3B8' },
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { 
            color: '#cbd5e1'
        },
        border: { display: false }
      },
      y: {
        grid: { color: 'rgba(0, 0, 0, 0.1)' }
      }
    }
  };

  return <div className="chart-container"><Bar data={data} options={options} /></div>;
};

const Recommendations = ({ recommendations }) => {
  if (!recommendations) {
    return (
      <div className="recommendation-item">
        <h4>💡 Персонализированные рекомендации</h4>
        <p>На основе прогнозируемого дохода клиента формируются индивидуальные предложения по финансовым продуктам Альфа-Банка</p>
      </div>
    );
  }

  // Check if it's a string (from API prediction) or object (from DB)
  let recText = "";
  if (typeof recommendations === 'string') {
      recText = recommendations;
  } else if (recommendations.recommendations) {
      recText = recommendations.recommendations;
  }

  const recs = recText.split('|').filter(r => r.trim());

  return (
    <div className="recommendations-list">
      {recs.map((rec, index) => (
        <div key={index} className="recommendation-item">
          <h4>💰 Предложение {index + 1}</h4>
          <p>{rec.trim()}</p>
        </div>
      ))}
    </div>
  );
};

function App() {
  const [mode, setMode] = useState('search'); // 'search' or 'calc'
  
  // Search State
  const [searchId, setSearchId] = useState('');
  const [foundClient, setFoundClient] = useState(null);
  
  // Calc State
  const [calcForm, setCalcForm] = useState({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advancedFilter, setAdvancedFilter] = useState('');
  const [calcResult, setCalcResult] = useState(null);

  // Search Autocomplete State
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Common State
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  
  const API_URL = 'http://localhost:8000/api';

  // Initial Data Load
  React.useEffect(() => {
    const initData = async () => {
      try {
        const featuresRes = await axios.get(`${API_URL}/features`);
        setCalcForm(featuresRes.data);
        setApiReady(true);
      } catch (err) {
        console.error("Failed to load features:", err);
      }
    };
    initData();
  }, []);

  // Close suggestions on click outside
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest('.search-input-group')) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const fetchSuggestions = async (query) => {
    try {
      const res = await axios.get(`${API_URL}/clients/search`, { params: { q: query } });
      setSuggestions(res.data);
    } catch (err) {
      console.error("Failed to fetch suggestions", err);
    }
  };

  const handleSearchChange = (e) => {
    const val = e.target.value.replace(/[^0-9]/g, '');
    setSearchId(val);
    fetchSuggestions(val);
    setShowSuggestions(true);
  };

  const handleSuggestionClick = (client) => {
    setSearchId(String(client.id));
    setShowSuggestions(false);
    // Immediate search
    loadClientData(String(client.id));
  };

  const loadClientData = async (id) => {
    setLoading(true);
    setError(null);
    setFoundClient(null);

    try {
      const response = await axios.get(`${API_URL}/client/${id}`);
      setFoundClient(response.data);
    } catch (err) {
      console.error("Search error:", err);
      if (err.response && err.response.status === 404) {
        setError('Клиент с указанным ID не найден в базе данных.');
      } else {
        setError('Произошла ошибка при поиске данных.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    if (!searchId.trim()) {
      setError('Пожалуйста, введите ID клиента');
      return;
    }
    loadClientData(searchId);
  };

  const handleCalculate = async () => {
    setLoading(true);
    setError(null);
    setCalcResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, calcForm);
      setCalcResult(response.data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError('Ошибка при расчете прогноза. Проверьте соединение с сервером.');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = useCallback((key, value) => {
    setCalcForm(prev => ({
      ...prev,
      [key]: value
    }));
  }, []);

  // Helper to check if a feature is a "Key Feature" (already shown prominently)
  const isKeyFeature = (key) => {
    return ['age', 'gender', 'salary_6to12m_avg', 'avg_cur_cr_turn'].includes(key);
  };

  return (
    <div className="container">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <h1>Сервис прогнозирования доходов</h1>
          </div>
          <p className="tagline">Точный прогноз — уверенные решения для Альфа-Банка</p>
        </div>
      </header>

      <main className="main-content">
        
        {/* Mode Switcher */}
        <div className="mode-switch-container">
          <button 
            onClick={() => { setMode('search'); setError(null); }}
            className={`mode-btn ${mode === 'search' ? 'active' : ''}`}
          >
            <Search size={20} />
            Поиск клиента
          </button>
          <button 
            onClick={() => { setMode('calc'); setError(null); }}
            disabled={!apiReady}
            className={`mode-btn ${mode === 'calc' ? 'active' : ''}`}
          >
            <Calculator size={20} />
            Калькулятор
          </button>
        </div>

        {/* SEARCH MODE */}
        {mode === 'search' && (
          <>
            <section className="search-section">
              <div className="search-card">
                <h2>Поиск клиента</h2>
                <p>Введите ID клиента для анализа дохода и получения персонализированных предложений</p>
                
                <div className="search-input-group">
                  <div className="input-wrapper">
                    <input 
                      type="text" 
                      value={searchId}
                      onChange={handleSearchChange}
                      onFocus={() => { fetchSuggestions(searchId); setShowSuggestions(true); }}
                      onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                      placeholder="Введите ID клиента..."
                      className="search-input"
                      autoComplete="off"
                    />
                    
                    {/* Suggestions Dropdown */}
                    {showSuggestions && suggestions.length > 0 && (
                      <div className="suggestions-dropdown">
                        <div className="suggestions-header">
                          Найденные клиенты
                        </div>
                        {suggestions.map(client => (
                          <div 
                            key={client.id}
                            onClick={() => handleSuggestionClick(client)}
                            className="suggestion-item"
                          >
                            <div className="client-info">
                               <div className="client-avatar">
                                 <User size={16} />
                               </div>
                               <span className="client-id">ID: {client.id}</span>
                            </div>
                            <div className="prediction-info">
                              <div className="prediction-label">Прогноз</div>
                              <div className="prediction-amount">{formatCurrency(client.predicted_income)}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <button 
                    onClick={handleSearch} 
                    className="search-button" 
                    disabled={loading}
                  >
                    {loading ? <Loader2 className="animate-spin" /> : <Search size={20} />}
                    <span>{loading ? '' : 'Найти'}</span>
                  </button>
                </div>
              </div>
            </section>

            {foundClient && (
              <section className="results-section animate-fade-in">
                <div className="client-header">
                  <h2>Клиент #{foundClient.id}</h2>
                  <div className="client-status">
                    <div className="status-badge active">Клиент найден</div>
                  </div>
                </div>

                <div className="prediction-card">
                  <div className="card-header">
                    <h3>Прогнозируемый доход</h3>
                    <div className="accuracy-badge">Точность модели: 94%</div>
                  </div>
                  <div className="income-display">
                    <div className="income-value">
                      {formatCurrency(foundClient.submission.target || foundClient.submission.predicted_income)}
                    </div>
                    <div className="income-label">рублей в месяц</div>
                  </div>
                </div>

                <div className="analysis-card">
                  <div className="card-header">
                    <h3>Факторы влияния на прогноз</h3>
                    <p className="card-subtitle">Наиболее значимые параметры, повлиявшие на расчет дохода</p>
                  </div>
                  <ShapChart shapData={foundClient.shap} />
                </div>

                <div className="recommendations-card">
                  <div className="card-header">
                    <h3>Персонализированные предложения</h3>
                    <p className="card-subtitle">Рекомендации на основе прогнозируемого дохода клиента</p>
                  </div>
                  <Recommendations recommendations={foundClient.recommendations} />
                </div>
              </section>
            )}
          </>
        )}

        {/* CALCULATOR MODE */}
        {mode === 'calc' && (
          <>
             <section className="search-section">
              <div className="search-card text-left max-w-4xl mx-auto">
                <div className="text-center mb-10">
                    <h2 className="text-3xl font-bold mb-2 text-gray-800">Калькулятор дохода</h2>
                    <p className="text-gray-500">Спрогнозируйте доход клиента на основе ключевых показателей</p>
                </div>
                
                {/* Key Parameters Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
                  
                  {/* Age Card */}
                  <div className="param-card">
                    <label className="param-label">
                        <User size={18} className="text-blue-500" />
                        Возраст
                    </label>
                    <input 
                      type="number" 
                      className="custom-input" 
                      value={calcForm.age || ''}
                      onChange={(e) => handleInputChange('age', e.target.value)}
                      placeholder="Например: 35"
                    />
                  </div>
                  
                  {/* Gender Card */}
                  <div className="param-card">
                    <label className="param-label">
                        <User size={18} className="text-pink-500" />
                        Пол
                    </label>
                    <select 
                      className="custom-input custom-select"
                      value={calcForm.gender || 'Мужской'}
                      onChange={(e) => handleInputChange('gender', e.target.value)}
                    >
                      <option value="Мужской">Мужской</option>
                      <option value="Женский">Женский</option>
                    </select>
                  </div>

                  {/* Salary Card */}
                  <div className="param-card">
                    <label className="param-label">
                        <Wallet size={18} className="text-green-500" />
                        Зарплата (6-12 мес)
                    </label>
                    <input 
                      type="number" 
                      className="custom-input" 
                      value={calcForm.salary_6to12m_avg || ''}
                      onChange={(e) => handleInputChange('salary_6to12m_avg', e.target.value)}
                      placeholder="0 ₽"
                    />
                  </div>

                  {/* Credit Turnover Card */}
                  <div className="param-card">
                    <label className="param-label">
                        <CreditCard size={18} className="text-purple-500" />
                        Оборот по кредитам
                    </label>
                    <input 
                      type="number" 
                      className="custom-input" 
                      value={calcForm.avg_cur_cr_turn || ''}
                      onChange={(e) => handleInputChange('avg_cur_cr_turn', e.target.value)}
                      placeholder="0 ₽"
                    />
                  </div>
                </div>

                {/* Advanced Parameters Toggle */}
                <div className="mb-8">
                   <button 
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="toggle-advanced-btn"
                   >
                     {showAdvanced ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                     {showAdvanced ? 'Скрыть дополнительные параметры' : `Показать все параметры (${Object.keys(calcForm).length})`}
                   </button>

                   {/* Advanced Parameters Grid */}
                   {showAdvanced && (
                      <div className="advanced-grid-container">
                         <div className="mb-6 relative">
                           <Search className="absolute left-3 top-3.5 text-gray-400" size={18} />
                           <input 
                             type="text" 
                             placeholder="Поиск по названию параметра..." 
                             className="w-full p-3 pl-10 border border-gray-200 rounded-xl focus:border-blue-500 outline-none"
                             value={advancedFilter}
                             onChange={(e) => setAdvancedFilter(e.target.value)}
                           />
                         </div>
                         
                         <div className="grid gap-4 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
                            {Object.entries(calcForm)
                              .filter(([key]) => !isKeyFeature(key)) 
                              .filter(([key]) => key.toLowerCase().includes(advancedFilter.toLowerCase()) || formatFeatureName(key).toLowerCase().includes(advancedFilter.toLowerCase()))
                              .slice(0, 50) 
                              .map(([key, value]) => (
                                <ParamInput 
                                  key={key} 
                                  paramKey={key} 
                                  value={value} 
                                  onChange={handleInputChange} 
                                />
                              ))
                            }
                            {/* Show message if truncated */}
                            {Object.entries(calcForm).filter(([key]) => !isKeyFeature(key)).length > 50 && (
                               <div className="col-span-full text-center py-4 text-gray-400 text-sm">
                                 Показаны первые 50 параметров. Используйте поиск, чтобы найти остальные.
                               </div>
                            )}
                         </div>
                      </div>
                   )}
                </div>

                <div className="flex justify-center calculate-btn-container">
                  <button 
                    onClick={handleCalculate} 
                    className="search-button w-full md:w-auto min-w-[200px] py-4 text-lg shadow-xl hover:shadow-2xl transform hover:-translate-y-1 transition-all" 
                    disabled={loading}
                  >
                    {loading ? <Loader2 className="animate-spin" /> : <Calculator size={24} />}
                    <span>{loading ? 'Вычисляем...' : 'Рассчитать прогноз'}</span>
                  </button>
                </div>
              </div>
            </section>

            {calcResult && (
               <section className="results-section animate-fade-in">
               <div className="prediction-card border-l-8 border-l-[#a9ef01]">
                 <div className="card-header">
                   <h3>Результат расчета</h3>
                   <div className="accuracy-badge bg-blue-500">Real-time Inference</div>
                 </div>
                 <div className="income-display">
                   <div className="income-value text-[#EF3124]">
                     {formatCurrency(calcResult.predicted_income)}
                   </div>
                   <div className="income-label">прогнозируемый доход</div>
                 </div>
               </div>

               <div className="analysis-card">
                  <div className="card-header">
                    <h3>Факторы влияния на прогноз</h3>
                    <p className="card-subtitle">Наиболее значимые параметры для данного прогноза</p>
                  </div>
                  <ShapChart shapData={calcResult.shap} />
                </div>

               <div className="recommendations-card">
                 <div className="card-header">
                   <h3>Рекомендации системы</h3>
                   <p className="card-subtitle">Сгенерировано автоматически на основе прогноза</p>
                 </div>
                 <Recommendations recommendations={calcResult.recommendations} />
               </div>
             </section>
            )}
          </>
        )}

        {error && (
          <section className="error-section">
            <div className="error-card">
              <div className="error-icon">⚠️</div>
              <h2>Ошибка</h2>
              <p>{error}</p>
              <button className="retry-button" onClick={() => setError(null)}>
                Закрыть
              </button>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Разработано командой "Титаник 2" для хакатона Альфа-Банка 2025</p>
      </footer>
    </div>
  );
}

export default App;