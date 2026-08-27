```bash
uv run python src/assistant/main.py

# или
uv run start-bot
```

Запуск
Модель в LM Studio должна быть загружена (qwen/qwen3.6-35b-a3b, сервер на localhost:1234).
```
# обычный прогон
.venv/bin/python -m assistant.main "Когда вышел Django 6.0?"

# с размышлением модели в логе
SHOW_REASONING=1 .venv/bin/python -m assistant.main "Когда вышел Django 6.0?"

# Через uv, если venv не активирован:
uv run python -m assistant.main "вопрос"

# Есть и entry point из pyproject.toml:
uv run start-bot "вопрос"
```