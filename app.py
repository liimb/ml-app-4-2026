import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import catboost as cb
import lightgbm as lgb

st.set_page_config(page_title="Diamond Price Predictor", layout="wide")

cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI1': 1, 'SI2': 2, 'VS1': 3, 'VS2': 4, 'VVS1': 5, 'VVS2': 6, 'IF': 7}
model_names = {
    "ML1: Полиномиальная регрессия": "model_ml1.pkl",
    "ML2: Gradient Boosting": "model_ml2.pkl",
    "ML3: CatBoost": "model_ml3.cbm",
    "ML4: LightGBM": "model_ml4.txt",
    "ML5: Stacking": "model_ml5.pkl",
    "ML6: Нейронная сеть": "model_ml6.keras"
}

@st.cache_resource
def load_resources():
    res = {}
    
    for key in ['ML1', 'ML2', 'ML5']:
        with open(f'models/model_{key.lower()}.pkl', 'rb') as f:
            res[key] = pickle.load(f)

    try:
        model_cb = cb.CatBoostRegressor()
        res['ML3'] = model_cb.load_model('models/model_ml3.cbm')
    except Exception as e:
        st.error(f"Ошибка CatBoost: {e}")

    try:
        res['ML4'] = lgb.Booster(model_file='models/model_ml4.txt')
    except Exception as e:
        st.error(f"Ошибка LightGBM: {e}")

    res['ML6'] = load_model('models/model_ml6.keras')
    with open('models/pca_transformer.pkl', 'rb') as f:
        res['pca'] = pickle.load(f)
        
    return res

resources = load_resources()

page = st.sidebar.radio("Навигация", ["Разработчик", "Датасет", "Визуализация", "Предсказание"])

if page == "Разработчик":
    st.title("О разработчике")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Теницкий Артемий Сергеевич")
        st.write("**Группа:** ФИТ-242")
        st.write("**Тема РГР:** Разработка Web-приложения для инференса моделей ML и анализа данных алмазов")

elif page == "Датасет":
    st.title("О наборе данных")
    st.write("Датасет содержит информацию о характеристиках алмазов.")
    st.markdown("""
    **Признаки:**
        1. `carat` - карат алмаза (масса) (от 0.2 до 5.01)
        2. `cut` - качество огранки (`Fair` - плохо, `Good`- хорошо, `Very Good`-очень хорошо, `Premium`-премиальная, `Ideal`-идеальная), ранжированный категориальный признак
        3. `color` - цвет алмаза (`J`-плохой, `I`, `H`, `G`, `F`, `E`, `D`-бесцветный), ранжированный категориальный признак
        4. `clarity` - чистота/прозрачность (`I1`-видны дефекты внутри,`SI1`,`SI2`,`VS1`,`VS2`,`VVS1`,`VVS2`,`IF`-идеальный внутри), ранжированный категориальный признак
        5. `depth` - глубина в процентах (отношение высоты к среднему диаметру) (z / mean(x, y) * 100 = ((2 * z) / (x + y)) * 100 ) (от 43 до 79)
        6. `table` - размер верхней плоской грани относительно ширины в процентах (от 43 до 95)
        7. `price` - (от 326 до 18823)
        8. `x` - длина в мм (от 0 до 10.74)
        9. `y` - ширина в мм (от 0 до 58.9)
        10. `z` - высота в мм (от 0 до 31.8)
    """)

    try:
        df_sample = pd.read_csv("data/diamonds_clean.csv")
        st.dataframe(df_sample.head(10))
    except:
        st.warning("Файл data/diamonds_clean.csv не найден для предпросмотра.")

elif page == "Визуализация":
    st.title("Визуализация зависимостей")
    df = pd.read_csv("data/diamonds_clean.csv")
    
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(df['price'], bins=30, kde=True, color='blue')
    plt.title("Распределение цен")

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='carat', y='price', alpha=0.5)
    plt.title("Зависимость цены от карат")

    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='cut', y='price')
    plt.title("Цена в зависимости от огранки")

    plt.subplot(2, 2, 4)
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='RdYlGn')
    plt.title("Матрица корреляции")

    st.pyplot(fig)

elif page == "Предсказание":
    st.title("Предсказание стоимости")
    
    selected_display_name = st.selectbox("Выберите архитектуру модели", list(model_names.keys()))
    
    model_key = selected_display_name.split(":")[0] 
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        carat = st.number_input("Вес (carat)", 0.2, 5.0, 0.7, step=0.01)
        cut = st.selectbox("Огранка (cut)", list(cut_map.keys()))
        color = st.selectbox("Цвет (color)", list(color_map.keys()))
        
    with col2:
        clarity = st.selectbox("Чистота (clarity)", list(clarity_map.keys()))
        depth = st.number_input("Глубина (depth %)", 40.0, 80.0, 61.7, step=0.1)
        table = st.number_input("Таблица (table)", 40.0, 100.0, 56.0, step=0.1)
        
    with col3:
        x = st.number_input("Длина x (mm)", 0.0, 15.0, 5.7, step=0.01)
        y = st.number_input("Ширина y (mm)", 0.0, 15.0, 5.7, step=0.01)
        z = st.number_input("Высота z (mm)", 0.0, 15.0, 3.5, step=0.01)

    if st.button("Рассчитать стоимость", use_container_width=True):
        features = np.array([[
            carat, 
            cut_map[cut], 
            color_map[color], 
            clarity_map[clarity], 
            depth, 
            table, 
            x, y, z
        ]])
        
        try:
            if model_key == "ML6":
                features_transformed = resources['pca'].transform(features)
                raw_pred = resources['ML6'].predict(features_transformed, verbose=0)
                prediction = raw_pred[0][0]
            else:
                prediction = resources[model_key].predict(features)[0]
            
            st.metric(label="Прогноз стоимости", value=f"${prediction:,.2f}")
            st.balloons()
            
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
