"""
Переменные окружения. Единственное место, где читается .env.

Модули берут значения отсюда, а не из os.getenv, чтобы список настроек
проекта был виден одним файлом.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from assistant.integrations.web import WebConfig

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Провайдер моделей ---------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")

LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "lm-studio")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/gemma-4-26b-a4b-qat")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.7-flash")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

YC_BASE_URL = os.getenv("YC_BASE_URL", "https://llm.api.cloud.yandex.net/v1")
YC_API_KEY = os.getenv("YC_API_KEY", "")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID", "")
YC_MODEL = os.getenv("YC_MODEL", "yandexgpt-lite")

LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "")
LLM_SEED = os.getenv("LLM_SEED", "")

# --- Веб-слой ------------------------------------------------------------
# Строка User-Agent для запросов к сайтам. Wikimedia и часть других площадок
# отвечают 403 на agent без контакта: в скобках должен стоять адрес проекта
# или почта, по которым владельца бота можно найти.
WEB_USER_AGENT = os.getenv(
    "WEB_USER_AGENT",
    "omni-assistant/0.1 (+https://github.com/samtakoy)",
)

WEB_TIMEOUT_SECONDS = 10.0
WEB_MAX_PAGE_BYTES = 2_000_000

# Каталог кеша страниц и поисковой выдачи. Пустое значение выключает кеш целиком.
WEB_CACHE_DIR = os.getenv("WEB_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "web")).strip()

# Срок годности записей в днях. Ноль - не протухают никогда, и это осознанный
# выбор: собранная выдача нужна для воспроизводимых прогонов, а забытая
# настройка не должна однажды молча выбросить собранный материал. Устаревшие
# записи убираются удалением каталога, а не сроком.
WEB_PAGE_CACHE_TTL_DAYS = int(os.getenv("WEB_PAGE_CACHE_TTL_DAYS", "0"))
WEB_SEARCH_CACHE_TTL_DAYS = int(os.getenv("WEB_SEARCH_CACHE_TTL_DAYS", "0"))

# Обход кеша на чтение: нужен, когда правятся формулировки поисковых запросов и
# замороженная выдача мешает увидеть результат правки. Запись продолжается,
# поэтому такой прогон обновляет хранилище.
WEB_CACHE_BYPASS = os.getenv("WEB_CACHE_BYPASS", "").strip().lower() in ("1", "true", "yes")



def _cache_directory(raw_path: str) -> Path | None:
    """
    Приводит значение WEB_CACHE_DIR к каталогу кеша.

    Относительный путь считается от корня проекта, а не от текущего каталога:
    иначе запуск из другой папки заводил бы себе отдельный кеш, и собранный
    материал выглядел бы потерянным.

    Аргументы:
        raw_path: значение переменной окружения.

    Возвращает:
        Каталог кеша либо None, если значение пустое и кеш выключен.
    """
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


WEB_CONFIG = WebConfig(
    user_agent = WEB_USER_AGENT,
    request_timeout_seconds = WEB_TIMEOUT_SECONDS,
    max_page_bytes = WEB_MAX_PAGE_BYTES,
    cache_directory = _cache_directory(raw_path = WEB_CACHE_DIR),
    page_cache_ttl_days = WEB_PAGE_CACHE_TTL_DAYS,
    search_cache_ttl_days = WEB_SEARCH_CACHE_TTL_DAYS,
    bypass_cache = WEB_CACHE_BYPASS,
)

# Отладочный режим: включить размышление в фазе поиска и показать его текст.
# Дороже примерно вчетверо по времени, поэтому по умолчанию выключен.
SHOW_REASONING = os.getenv("SHOW_REASONING", "").strip().lower() in ("1", "true", "yes")


