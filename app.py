import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

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
        st.write("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных")

elif page == "Датасет":
    st.title("О наборе данных")
    st.write("Датасет содержит информацию о характеристиках алмазов. Необходимо на основе 9 различных признаков алмаза предсказать его возможную стоимость")
    st.markdown("""
    **Описание всех признаков:**
                
    1. `carat` - карат алмаза (масса) (от 0.2 до 5.01);
            
    2. `cut` - качество огранки (`Fair` - плохо, `Good`- хорошо, `Very Good`-очень хорошо, `Premium`-премиальная, `Ideal`-идеальная), ранжированный категориальный признак;
                
    3. `color` - цвет алмаза (`J`-плохой, `I`, `H`, `G`, `F`, `E`, `D`-бесцветный), ранжированный категориальный признак;
                
    4. `clarity` - чистота/прозрачность (`I1`-видны дефекты внутри,`SI1`,`SI2`,`VS1`,`VS2`,`VVS1`,`VVS2`,`IF`-идеальный внутри), ранжированный категориальный признак;
                
    5. `depth` - глубина в процентах (отношение высоты к среднему диаметру) (z / mean(x, y) * 100 = ((2 * z) / (x + y)) * 100 ) (от 43 до 79);
                
    6. `table` - размер верхней плоской грани относительно ширины в процентах (от 43 до 95);
                
    7. `price` - (от 326 до 18823);
                
    8. `x` - длина в мм (от 0 до 10.74);
                
    9. `y` - ширина в мм (от 0 до 58.9);
                
    10. `z` - высота в мм (от 0 до 31.8).
                
    """)
    st.subheader('Особенности EDA')
    st.markdown("""
    При обработке датасета были выполнены следующие задачи:
                
    - удален лишний столбец с индексом;
                
    - удалены дублирующиеся записи;
                
    - исправлены ошибочные выбросы и некоторые зависящие друг от друга признаки (x, y, z, depth, table);
    
    - закодированы категориальные признаки в соответствии с данными словарями:
    """)

    st.code("""
    cut_map = {
            'Fair': 0, 
            'Good': 1, 
            'Very Good': 2, 
            'Premium': 3, 
            'Ideal': 4
    }
            
    color_map = {
            'J': 0, 
            'I': 1, 
            'H': 2, 
            'G': 3, 
            'F': 4, 
            'E': 5, 
            'D': 6
    }
            
    clarity_map = {
            'I1': 0, 
            'SI1': 1, 
            'SI2': 2, 
            'VS1': 3, 
            'VS2': 4, 
            'VVS1': 5, 
            'VVS2': 6, 
            'IF': 7
    }
""", 
language='python')
    
    st.subheader('Получившийся датасет:')

    try:
        df_sample = pd.read_csv("data/diamonds_clean.csv")
        st.dataframe(df_sample.head(10))
        st.write("Всего записей:")
        st.write(len(df_sample))
        st.subheader('Метод describe():')
        st.dataframe(df_sample.describe())
    except:
        st.warning("Файл data/diamonds_clean.csv не найден для предпросмотра.")

elif page == "Визуализация":
    st.title("Исследование данных")
    st.write("Используйте вкладки ниже, чтобы просмотреть зависимости")

    df = pd.read_csv("data/diamonds_clean.csv")

    tab1, tab2, tab3 = st.tabs(["Цены и распределение", "Параметры и цена", "Корреляция признаков"])

    with tab1:
        st.subheader("Анализ распределения стоимости")

        fig_price = px.histogram(
            df, x="price", nbins=50, 
            marginal="box", 
            title="Распределение цен алмазов",
            color_discrete_sequence=['#00CC96'],
            labels={'price': 'Цена ($)'},
            template="plotly_white"
        )

        fig_price.update_layout(width=600, height=600, autosize=True)

        st.plotly_chart(fig_price)
        
    with tab2:
        st.subheader("Влияние характеристик на цену")
        
        feature = st.selectbox("Выберите характеристикy для анализа:", ["carat", "cut", "color", "clarity"])
        
        if feature == "carat":
            fig = px.scatter(
                df.sample(min(len(df), 5000)), 
                x="carat", y="price", color="clarity",
                size="depth", hover_data=['x', 'y', 'z'],
                title="Связь массы и стоимости",
                opacity=0.5
            )
            fig.update_layout(width=600, height=600, autosize=True)
        else:
            fig = px.box(
                df, x=feature, y="price", color=feature,
                title=f"Разброс цен в зависимости от {feature}",
                notched=True
            )
            fig.update_layout(width=600, height=600, autosize=True)
        
        st.plotly_chart(fig)

    with tab3:
        st.subheader("Корреляция непрерывных признаков")
        
        cols_to_corr = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
        corr = df[cols_to_corr].corr().round(2)
        
        fig_corr = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale='magenta',
            showscale=True
        )
        fig_corr.update_layout(width=600, height=600, autosize=False)
        st.plotly_chart(fig_corr)

elif page == "Предсказание":
    st.title("Предсказание стоимости")
    
    selected_display_name = st.selectbox("Выберите архитектуру модели", list(model_names.keys()))
    model_key = selected_display_name.split(":")[0] 
    
    mode = st.radio("Метод ввода данных:", ["Одиночное предсказание", "Загрузка файла (CSV)"], horizontal=True)

    if mode == "Одиночное предсказание":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            carat = st.number_input("Вес (carat)", 0.01, 300.0, 0.7, step=0.01)
            cut = st.selectbox("Огранка (cut)", list(cut_map.keys()))
            color = st.selectbox("Цвет (color)", list(color_map.keys()))
            
        with col2:
            clarity = st.selectbox("Чистота (clarity)", list(clarity_map.keys()))
            depth = st.number_input("Глубина (depth %)", 0.01, 100.0, 61.7, step=0.1)
            table = st.number_input("Верхняя плоскость (table %)", 0.01, 100.0, 56.0, step=0.1)
            
        with col3:
            x = st.number_input("Длина x (mm)", 0.01, 100.0, 5.7, step=0.01)
            y = st.number_input("Ширина y (mm)", 0.01, 100.0, 5.7, step=0.01)
            z = st.number_input("Высота z (mm)", 0.01, 100.0, 3.5, step=0.01)

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

    else:
        st.subheader("Множественное предсказание из CSV")
        st.write("Загрузите файл с признаками: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`")
        
        uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Предпросмотр данных:")
            st.dataframe(input_df.head())
            st.write("Всего записей:")
            st.write(len(input_df))

            if st.button("Запустить предсказание для всех строк", type="primary", use_container_width=True):
                try:
                    temp_df = input_df.copy()
        
                    maps = {
                        'cut': cut_map,
                        'color': color_map,
                        'clarity': clarity_map
                    }
                    
                    for col, mapping in maps.items():
                        if temp_df[col].dtype == 'object':
                            temp_df[col] = temp_df[col].map(mapping)
                    
                    if temp_df[['cut', 'color', 'clarity']].isnull().values.any():
                        st.error("Ошибка: В категориях есть неизвестные значения. Проверьте правильность написания (например, 'Ideal', 'Premium').")
                        st.stop()

                    feature_cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
                    X = temp_df[feature_cols].values
                    
                    if model_key == "ML6":
                        X_transformed = resources['pca'].transform(X)
                        predictions = resources['ML6'].predict(X_transformed, verbose=0).flatten()
                    else:
                        predictions = resources[model_key].predict(X)
                    
                    input_df['predicted_price'] = np.round(predictions, 2)
                    
                    st.success("Расчет окончен")
                    st.dataframe(input_df)
                    
                    output_csv = input_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Скачать файл с предсказаниями",
                        data=output_csv,
                        file_name="diamonds_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Ошибка при обработке файла: {e}. Проверьте названия колонок.")
