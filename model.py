import streamlit as st
from generate import generate_response

WELCOME_MESSAGE = """
### Добро пожаловать в Ассистент-Косметолог!
Здесь вы можете получить рекомендации по уходу за кожей, подбору косметических средств и определению типа кожи.
"""

FIRST_MESSAGE = """
Я помогу вам определить тип кожи, выявить возможные проблемы (сухость, жирность, чувствительность, акне) и подберу подходящие косметические средства.  
**Для лучшего результата укажите:**  
- Ваш возраст  
- Особенности кожи (если знаете)  
- Образ жизни (стресс, питание, вредные привычки)  
- Возможные аллергии
"""

st.title("Ассистент-косметолог")
st.info(WELCOME_MESSAGE)

# Инициализируем историю сообщений в session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            'role': 'system',
            'content': (
                'Отвечай на русском языке. Ты – опытный косметолог с многолетней практикой в уходе за кожей. '
                'Твоя задача – помогать пользователям определять их тип кожи, выявлять проблемы (например, сухость, жирность, чувствительность, акне) '
                'и давать персонализированные рекомендации по уходу и выбору косметических средств. Если информации недостаточно, уточняй детали '
                '(возраст, образ жизни, наличие аллергий). '
                'Ты также должен провести пользователя по сценарию “сбора корзины”, объяснить результат подбора и корректировать его на основе комментариев.'
                'тебе дают описание фотографии (оно поступает от другой нейросети, не нужно про нее упоминать никогда), твоя задача подчеркнуть важные детали и дать рекомендации по уходу за кожей. объяснить свой выбор'
            )
        },
        {
            'role': 'assistant',
            'content': FIRST_MESSAGE
        }
    ]

def main():
    # Отображение истории диалога в обратном порядке (новые сообщения сверху)
    with st.container():
        for message in reversed(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            if role == "user":
                # Если в сообщении присутствует описание изображения, показываем только часть до него.
                if "Описание изображения:" in content:
                    content = content.split("Описание изображения:")[0].strip()
                st.markdown(f"**👤 Вы:** {content}")
            elif role == "assistant":
                st.markdown(f"**🧑‍⚕️ Ассистент:**\n\n{content}")

    # Поле для ввода текстового запроса
    user_input = st.text_area("Введите ваш запрос:")
    
    # Виджет для загрузки изображения (опционально)
    uploaded_image = st.file_uploader("Загрузите изображение (опционально)", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Загруженное изображение", use_container_width=True)
    
    if st.button("Получить рекомендации"):
        if user_input.strip() or uploaded_image is not None:
            with st.spinner("Модель обрабатывает запрос..."):
                image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
                response = generate_response(user_input, st.session_state.messages, image_bytes)
                st.markdown(f"**🧑‍⚕️ Ассистент:**\n\n{response}")
        else:
            st.warning("Пожалуйста, введите текст запроса или загрузите изображение.")

if __name__ == "__main__":
    main()
