import streamlit as st
import requests

# URL FastAPI-сервиса
API_URL = "http://127.0.0.1:8000"

st.title("Управление ML моделями")

# Выбор действия
st.sidebar.title("Навигация")
action = st.sidebar.selectbox("Выберите действие", ["Обучение модели", "Просмотр моделей", "Предсказание", "Удаление модели"])

# Функция для обучения модели
if action == "Обучение модели":
    st.header("Обучение модели")
    model_type = st.selectbox("Тип модели", ["logistic", "random_forest"])
    params = {}
    if model_type == "logistic":
        params["max_iter"] = st.number_input("Количество итераций (max_iter)", min_value=100, value=100)
    elif model_type == "random_forest":
        params["n_estimators"] = st.number_input("Число деревьев (n_estimators)", min_value=10, value=10)
        params["max_depth"] = st.number_input("Максимальная глубина (max_depth)", min_value=1, value=5)

    if st.button("Начать обучение"):
        # Изменено на отправку параметров запроса вместо JSON-тела
        response = requests.post(f"{API_URL}/train/?model_type={model_type}", json=params)
        if response.status_code == 200:
            st.success(f"Модель успешно обучена с ID: {response.json()['model_id']}")
        else:
            st.error(f"Ошибка: {response.json()['detail']}")

# Функция для просмотра моделей
elif action == "Просмотр моделей":
    st.header("Доступные модели")
    response = requests.get(f"{API_URL}/models/")
    if response.status_code == 200:
        models = response.json()
        if models:
            for model in models:
                st.write(f"ID: {model['id']}")
        else:
            st.write("Нет доступных моделей.")
    else:
        st.error(f"Ошибка: {response.json()['detail']}")

# Функция для предсказания
elif action == "Предсказание":
    st.header("Предсказание")
    model_id = st.number_input("ID модели", min_value=1, step=1)
    data = st.text_input("Введите данные (через запятую, например: 0.1,0.2,0.3,0.4,0.5)")
    data_list = [float(x) for x in data.split(",") if x]

    if st.button("Сделать предсказание"):
        response = requests.post(f"{API_URL}/predict/", json={"model_id": model_id, "data": data_list})
        if response.status_code == 200:
            st.success(f"Предсказание: {response.json()['prediction']}")
        else:
            st.error(f"Ошибка: {response.json()['detail']}")

# Функция для удаления модели
elif action == "Удаление модели":
    st.header("Удаление модели")
    model_id = st.number_input("ID модели для удаления", min_value=1, step=1)
    if st.button("Удалить модель"):
        response = requests.delete(f"{API_URL}/delete/", params={"model_id": model_id})
        if response.status_code == 200:
            st.success("Модель успешно удалена.")
        else:
            st.error(f"Ошибка: {response.json()['detail']}")
