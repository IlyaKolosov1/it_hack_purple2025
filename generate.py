# generate.py

import base64
from groq import Groq
from utils import trim_history

def generate_image_description(image_bytes):
    """
    Отправляет изображение с запросом "Опиши, что изображено на этом фото?" 
    в модель vision для получения текстового описания изображения.
    """
    client = Groq()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    message_content = [
        {"type": "text", "text": "Опиши, что изображено на этом фото?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": message_content}],
        model="llama-3.2-11b-vision-preview",
    )
    description = response.choices[0].message.content
    return description

def generate_response(answer: str, messages, image_bytes=None):
    """
    Генерирует ответ, объединяя текстовый запрос пользователя с описанием изображения (если оно предоставлено).
    Для запроса в модель используется объединённый текст, а в истории для отображения сохраняется только исходный запрос.
    """
    client = Groq()
    
    if image_bytes is not None:
        # Получаем описание изображения
        image_description = generate_image_description(image_bytes)
        # Объединяем пользовательский запрос с описанием изображения для отправки в модель
        combined_answer = f"{answer}\n\nОписание изображения: {image_description}"
    else:
        combined_answer = answer

    # Добавляем в историю полное сообщение для контекста
    messages.append({'role': 'user', 'content': combined_answer})
    
    trimmed_messages = trim_history(messages, max_tokens=6000)
    
    stream = client.chat.completions.create(
        messages=trimmed_messages,
        model='llama-3.3-70b-versatile',
        temperature=0.5,
        top_p=1,
        stop=None,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            full_response += content

    messages.append({'role': 'assistant', 'content': full_response})
    
    return full_response
