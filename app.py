import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import urllib.parse # Библиотека для создания ссылок

# --- Настройки страницы ---
st.set_page_config(page_title="Тим Бёртон Ассистент", page_icon="🦇", layout="wide")

# --- Функция для загрузки CSS стилей ---
def local_css(file_name):
    """Загружает внешний CSS-файл для стилизации приложения."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Критическая ошибка: не найден файл стилей '{file_name}'!")

# Применяем стили из файла style.css
local_css("style.css")

# --- Получение API ключа из секретов Streamlit ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Функция для загрузки и форматирования базы знаний ---
@st.cache_data
def create_knowledge_base():
    """Читает CSV-файл и возвращает DataFrame."""
    try:
        try:
            works_df = pd.read_csv("tim_burton_data.csv", sep=',').astype(str).fillna('не указано')
        except:
            works_df = pd.read_csv("tim_burton_data.csv", sep=';').astype(str).fillna('не указано')
        return works_df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None

# === Начало интерфейса приложения ===

st.title("🦇 Тим Бёртон Ассистент")
st.markdown("---")

# --- Поле для ввода текста ---
user_query = st.text_input(
    label=" ",
    placeholder="Спросите меня о фильмах, персонажах, стиле Тима Бёртона...",
    key="user_input_box",
    label_visibility="collapsed"
)

ask_button = st.button("**НАЙТИ ОТВЕТ**", use_container_width=True, key="find_answer")

# Загружаем базу знаний
works_dataframe = create_knowledge_base()
answer_placeholder = st.empty()

# Проверяем, что все готово к работе
if works_dataframe is not None and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "llama-3.1-8b-instant"
    except Exception as e:
        st.error(f"Ошибка инициализации клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner(""):
            st.markdown("<div class='spinner-text'>✨ Погружаюсь в атмосферу Бёртона...</div>", unsafe_allow_html=True)
            try:
                # Подготовка данных
                knowledge_base_text_for_model = ""
                for _, work in works_dataframe.iterrows():
                    knowledge_base_text_for_model += "-----\n"
                    knowledge_base_text_for_model += f"Название: {work['Name']}\n"
                    knowledge_base_text_for_model += f"Бюджет: {work.get('Budget', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Возрастной рейтинг: {work.get('Age rating', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Год выпуска: {work.get('Release year', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Сборы: {work.get('Box office', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Оригинальное название: {work.get('Original title', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Краткое описание: {work.get('Synopsis', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Продолжительность: {work.get('Duration', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Слоган: {work.get('Tagline', 'не указано')}\n"
                    knowledge_base_text_for_model += f"Страна: {work.get('Country', 'не указано')}\n"

                # Промпт
                prompt = f"""Твоя роль - быть экспертом по творчеству Тима Бёртона. Ты должен отвечать на вопросы, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленных данных.

СТРОГИЕ ИНСТРУКЦИИ:
1.  **ПОЛНЫЙ ПОИСК:** Найди ВСЕ записи, которые соответствуют запросу пользователя.
2.  **ПОЛНАЯ ИНФОРМАЦИЯ:** В блоке [РАССУЖДЕНИЯ] покажи ВСЕ найденные фильмы с ПОЛНОЙ информацией о каждом.
3.  **НИКАКИХ ДОГАДОК:** Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленных данных.
4.  **ФИЛЬТР ТЕМЫ:** Если запрос пользователя НЕ КАСАЕТСЯ актёров, фильмов, Тима Бёртона, его жанров, композиторов или персонажей его фильмов, ИЛИ если ответа нет в предоставленных данных, твой ответ должен быть СТРОГО одной фразой: "Извините, такого нет в базе, попробуйте поискать в интернете".
5.  **ФОРМАТ ОТВЕТА (ЕСЛИ ИНФОРМАЦИЯ НАЙДЕНА):** 
    [РАССУЖДЕНИЯ]
    ПОИСКОВЫЕ РЕЗУЛЬТАТЫ:
    
    🎬 [Название фильма 1]:
    🎭 Название: [полное название]
    💰 Бюджет: [бюджет]
    🔞 Возрастной рейтинг: [рейтинг]
    📅 Год выпуска: [год]
    🎫 Сборы: [сборы]
    🌎 Оригинальное название: [оригинал]
    📖 Описание: [описание]
    ⏱️ Продолжительность: [время]
    💬 Слоган: [слоган]
    🏴 Страна: [страна]
    
    АНАЛИЗ: [краткий анализ]
    
    [ОТВЕТ]
    [итоговый ответ пользователю]

ДАННЫЕ:
{knowledge_base_text_for_model}

ВОПРОС: {user_query}

ОТВЕТ:"""

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=3000
                )
                answer = response.choices[0].message.content

                # === ОБРАБОТКА ОТВЕТА ===
                
                # 1. Проверяем, вернула ли модель фразу об отсутствии данных
                if "Извините, такого нет в базе" in answer:
                    # Создаем ссылку на Google
                    encoded_query = urllib.parse.quote(user_query)
                    google_search_url = f"https://www.google.com/search?q={encoded_query}"
                    
                    full_response_html = f"""
                    <div style="text-align: center; padding: 20px; background-color: #2b2b2b; border-radius: 10px; border: 1px solid #ff6b6b; margin-top: 20px;">
                        <h3 style="color: #ff6b6b;">🦇 Извините, такого нет в базе</h3>
                        <p style="color: #cccccc; font-size: 1.1em;">{answer}</p>
                        <br>
                        <a href="{google_search_url}" target="_blank" style="text-decoration: none;">
                            <div style="
                                display: inline-block;
                                background-color: #4285F4;
                                color: white;
                                padding: 12px 24px;
                                border-radius: 5px;
                                font-weight: bold;
                                font-size: 16px;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                                transition: 0.3s;
                                ">
                                🔍 Найти ответ в Google
                            </div>
                        </a>
                    </div>
                    """
                
                # 2. Если ответ нормальный, обрабатываем формат [РАССУЖДЕНИЯ]...
                else:
                    try:
                        reasoning_part, final_answer_part = answer.split("[ОТВЕТ]")
                        reasoning_text = reasoning_part.replace("[РАССУЖДЕНИЯ]", "").strip()
                        final_answer_text = final_answer_part.strip()
                        
                        reasoning_html = reasoning_text.replace('\n', '<br>')
                        reasoning_html = reasoning_html.replace('🎬', '<span style="font-size: 1.3em;">🎬</span>')
                        final_answer_html = final_answer_text.replace('\n', '<br>')
                        final_answer_html = final_answer_html.replace('🎬', '<span style="font-size: 1.2em;">🎬</span>')

                        full_response_html = f"""
                        <div class='reasoning-section'>
                        <h3 style='color: #f0e68c; text-align: center;'>🔍 Результаты поиска:</h3>
                        <div class='films-list'>
                        {reasoning_html}
                        </div>
                        </div>
                        <br>
                        <div style='border-top: 2px solid #f0e68c; margin: 20px 0;'></div>
                        <br>
                        <div class='final-answer-section'>
                        <h3 style='color: #f0e68c; text-align: center;'>📋 Итоговый ответ:</h3>
                        <div class='final-answer'>
                        {final_answer_html}
                        </div>
                        </div>
                        """
                    except ValueError:
                        # Если формат нарушен, выводим просто текст
                        full_response_html = f'<div class="answer-text">{answer.replace(chr(10), "<br>")}</div>'

                # Выводим результат
                answer_placeholder.markdown(full_response_html, unsafe_allow_html=True)

            except Exception as e:
                answer_placeholder.markdown(f'<div class="error-message">🎃 Произошла ошибка: {e}</div>', unsafe_allow_html=True)
    
    elif not user_query and ask_button:
        answer_placeholder.markdown('<div class="warning-message">❓ Пожалуйста, введите ваш вопрос!</div>', unsafe_allow_html=True)

elif not works_dataframe:
    answer_placeholder.markdown('<div class="error-message">💀 Критическая ошибка: Не удалось загрузить базу знаний.</div>', unsafe_allow_html=True)
elif not GROQ_API_KEY:
    answer_placeholder.markdown('<div class="error-message">🔑 Ошибка API: Не установлен ключ GROQ.</div>', unsafe_allow_html=True)

# --- Дополнительная информация в сайдбаре (ВОССТАНОВЛЕНО) ---
with st.sidebar:
    st.markdown("### 💡 Примеры запросов:")
    st.markdown("""
    - **Фильмы с рейтингом 18+**
    - **Самые дорогие фильмы**  
    - **Фильмы 90-х годов**
    - **Фильмы с Джонни Деппом**
    - **Фильмы ужасов**
    - **Фильмы с самым высоким бюджетом**
    - **Фильмы выпущенные после 2000 года**
    """)
    
    st.markdown("### 📊 О базе данных:")
    if works_dataframe is not None:
        st.write(f"Всего произведений: **{len(works_dataframe)}**")
        st.write(f"Годы: **{works_dataframe['Release year'].min()} - {works_dataframe['Release year'].max()}**")
    
    st.markdown("---")
    st.markdown("### 🦇 О Тиме Бёртоне")
    st.markdown("""
    Тим Бёртон - американский режиссёр, продюсер и мультипликатор, 
    известный своим уникальным готическим стилем и сюрреалистичными 
    произведениями.
    """)
    
    st.markdown("---")
    if st.button("⬅️ Назад", use_container_width=True, key="back_main"):
        st.markdown("""
        <div style='background-color: #2b2b2b; padding: 15px; border-radius: 10px; border: 1px solid #f0e68c;'>
            <h4 style='color: #f0e68c; margin-top: 0;'>Перейти на главную страницу</h4>
            <a href='https://quixotic-shrimp-ea9.notion.site/9aabb68bd7004965819318e32d8ff06e?v=2b4a0ca7844a80d6aa8a000c6a7e5272' target='_blank' style='color: #ff6b6b; font-weight: bold;'>🏠 Главная страница проекта</a>
        </div>
        """, unsafe_allow_html=True)
