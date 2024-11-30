import streamlit as st
import requests
import json

# Настройки Streamlit
st.set_page_config(
    page_title="Model Management Dashboard",
    layout="wide",
)

# Основной заголовок
st.title("Model Management Service Dashboard")
st.markdown(
    "Управляйте обучением, предсказаниями и мониторингом моделей через веб-интерфейс."
)

# Базовый URL API
BASE_URL = "http://localhost:8000"


# Функция для получения списка моделей
def get_models():
    try:
        response = requests.get(f"{BASE_URL}/models/")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Ошибка при получении списка моделей.")
            return []
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")
        return []


# Функция для получения статуса сервиса
def get_status():
    try:
        response = requests.get(f"{BASE_URL}/status/")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Ошибка при получении статуса.")
            return {}
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")
        return {}


# Дашборд: Статус сервиса
st.header("Системный статус")
status = get_status()

if status:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Состояние", status.get("status", "Unknown"))
        st.metric("Использование памяти (%)", status["memory_usage"]["percent"])
    with col2:
        st.metric(
            "Всего памяти", f"{status['memory_usage']['total'] / (1024**3):.2f} GB"
        )
        st.metric(
            "Доступно памяти",
            f"{status['memory_usage']['available'] / (1024**3):.2f} GB",
        )
    st.write(
        "Метрика бизнес-логики:", status.get("business_logic_metric", "Нет данных")
    )
    st.write("Время работы системы:", status.get("uptime", "Нет данных"))

# Дашборд: Список моделей
st.header("Доступные модели")
models = get_models()

if models:
    for model in models:
        with st.expander(f"Модель ID: {model['id']}"):
            st.write("")
            # st.write(f"Параметры: {json.dumps(model['params'], indent=2)}")

# Обучение модели
st.header("Обучение новой модели")
with st.form("train_model_form"):
    model_type = st.text_input("Тип модели")
    params = st.text_area("Параметры модели (JSON)")
    X_train = st.text_area("Данные для обучения (X_train)")
    y_train = st.text_area("Данные для обучения (y_train)")
    submitted = st.form_submit_button("Запустить обучение")

    if submitted:
        try:
            payload = {
                "model_type": model_type,
                "params": json.loads(params),
                "X_train": json.loads(X_train),
                "y_train": json.loads(y_train),
            }
            response = requests.post(f"{BASE_URL}/train/", json=payload)
            if response.status_code == 200:
                st.success(f"Модель успешно обучена! ID: {response.json()['model_id']}")
            else:
                st.error(
                    f"Ошибка обучения: {response.json().get('detail', 'Неизвестная ошибка')}"
                )
        except Exception as e:
            st.error(f"Ошибка отправки запроса: {e}")

# Предсказание
st.header("Предсказание")
with st.form("predict_form"):
    model_id = st.number_input("ID модели", min_value=1, step=1)
    data = st.text_input("Данные для предсказания (через пробел)")
    predict_submitted = st.form_submit_button("Получить предсказание")

    if predict_submitted:
        try:
            response = requests.post(
                f"{BASE_URL}/predict/", params={"model_id": model_id, "data": data}
            )
            if response.status_code == 200:
                st.success(f"Результат предсказания: {response.json()['prediction']}")
            else:
                st.error(
                    f"Ошибка предсказания: {response.json().get('detail', 'Неизвестная ошибка')}"
                )
        except Exception as e:
            st.error(f"Ошибка отправки запроса: {e}")


# Удаление модели
st.header("Удаление модели")
with st.form("delete_form"):
    delete_model_id = st.number_input("ID модели для удаления", min_value=1, step=1)
    delete_submitted = st.form_submit_button("Удалить модель")

    if delete_submitted:
        try:
            response = requests.delete(
                f"{BASE_URL}/delete/", params={"model_id": delete_model_id}
            )
            if response.status_code == 200:
                st.success("Модель успешно удалена.")
            else:
                st.error(
                    f"Ошибка удаления: {response.json().get('detail', 'Неизвестная ошибка')}"
                )
        except Exception as e:
            st.error(f"Ошибка отправки запроса: {e}")
