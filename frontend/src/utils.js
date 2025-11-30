export const formatCurrency = (amount) => {
  if (isNaN(amount)) return '0 ₽';
  return new Intl.NumberFormat('ru-RU', {
    style: 'currency',
    currency: 'RUB',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
};

export const formatFeatureName = (featureName) => {
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
