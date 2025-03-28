# utils.py

def count_tokens(text: str) -> int:
    # Грубая оценка: считаем, что слово примерно равно 1 токену.
    return len(text.split())

def estimate_message_tokens(message) -> int:
    if isinstance(message["content"], str):
        return count_tokens(message["content"])
    elif isinstance(message["content"], list):
        tokens = 0
        for part in message["content"]:
            if part.get("type") == "text":
                tokens += count_tokens(part.get("text"))
        return tokens
    return 0

def trim_history(messages, max_tokens=6000):
    """
    Оставляет последние сообщения из истории так, чтобы суммарное количество токенов не превышало max_tokens.
    Сначала пробегаемся по истории с конца (самые последние сообщения), затем возвращаем обрезанную историю.
    """
    total_tokens = 0
    trimmed = []
    for message in reversed(messages):
        message_tokens = estimate_message_tokens(message)
        if total_tokens + message_tokens > max_tokens:
            break
        total_tokens += message_tokens
        trimmed.insert(0, message)
    return trimmed
