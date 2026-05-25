# Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных (Diamond Price Predictor)

В рамках проекта разработан интерактивный веб-дашборд на базе фреймворка **Streamlit**, предназначенный для пакетного/одиночного инференса шести предиктивных моделей для оценки стоимости алмазов.

## Ссылки на проект
* **Деплой приложения (Streamlit Cloud):** [Diamond Price Predictor](https://ml-app-4-2026-afzkk5wdzjowruedyrfwpu.streamlit.app/)

---

## Стек технологий и библиотека

* **Инфраструктура:** Python, Streamlit;
* **Анализ данных и визуализация:** Pandas, NumPy, Plotly, Seaborn, Matplotlib;
* **Машинное обучение (Классические алгоритмы и ансамбли):** Scikit-learn, CatBoost, LightGBM;
* **Глубокое обучение:** TensorFlow / Keras (Многослойный перцептрон).

---

## Исследуемые модели и метрики

| Идентификатор | Архитектура модели | Метрика качества ($R^2$) | Формат сериализации |
| :--- | :--- | :---: | :--- |
| **ML1** | Полиномиальная регрессия | `0.951` | `.pkl` (pickle) |
| **ML2** | Gradient Boosting Regressor | `0.981` | `.pkl` (pickle) |
| **ML3** | CatBoost Regressor | **`0.982`** | `.cbm` (native) |
| **ML4** | LGBMRegressor (LightGBM) | `0.980` | `.txt` (booster) |
| **ML5** | Stacking Regressor (Tree + Poly $\rightarrow$ Ridge) | `0.976` | `.pkl` (pickle) |
| **ML6** | Глубокая полносвязная нейронная сеть (FCNN + PCA) | `0.978` | `.keras` (native) |

---

## Структура проекта

```text
├── data/
│   └── diamonds_clean.csv       # Очищенный датасет после этапа EDA
├── models/
│   ├── model_ml1.pkl            # Веса полиномиальной регрессии
│   ├── model_ml2.pkl            # Веса Gradient Boosting
│   ├── model_ml3.cbm            # Архитектура и веса CatBoost
│   ├── model_ml4.txt            # Конфигурация Booster LightGBM
│   ├── model_ml5.pkl            # Стэкинг-ансамбль
│   ├── model_ml6.keras          # Полносвязная нейросеть Keras
│   └── pca_transformer.pkl      # Экспортированный объект PCA для ML6
├── app.py                       # Главный скрипт Streamlit-приложения
├── requirements.txt             # Зависимости проекта
└── README.md                    # Документация проекта
