"""
Переменные окружения. Единственное место, где читается .env.

Модули берут значения отсюда, а не из os.getenv, чтобы список настроек
проекта был виден одним файлом.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from assistant.integrations.listening import ListeningConfig
from assistant.integrations.web import WebConfig

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(raw_path: str) -> Path:
    """
    Приводит значение переменной окружения к пути внутри проекта.
    Аргументы:
        raw_path: значение переменной окружения.
    Возвращает:
        Путь к каталогу.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _cache_directory(raw_path: str) -> Path | None:
    """
    Приводит значение переменной окружения к каталогу кеша.

    Аргументы:
        raw_path: значение переменной окружения.

    Возвращает:
        Каталог кеша либо None, если значение пустое и кеш выключен.
    """
    if not raw_path:
        return None

    return _project_path(raw_path = raw_path)


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

# --- Модель зрения -------------------------------------------------------
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "").strip() or LLM_PROVIDER
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3-vl-4b").strip()

# До какого размера ужимать картинку перед отправкой.
VISION_MAX_SIDE = int(os.getenv("VISION_MAX_SIDE", "1024"))

# Качество пережатия.
VISION_JPEG_QUALITY = int(os.getenv("VISION_JPEG_QUALITY", "85"))

# Каталог кеша разборов картинок. Пустое значение выключает кеш.
VISION_CACHE_DIR = _cache_directory(
    raw_path = os.getenv("VISION_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "vision")).strip()
)
VISION_CACHE_TTL_DAYS = int(os.getenv("VISION_CACHE_TTL_DAYS", "0"))
VISION_CACHE_BYPASS = os.getenv("VISION_CACHE_BYPASS", "").strip().lower() in ("1", "true", "yes")

# --- Рассказчик ----------------------------------------------------------
# Способ сборки рассказчика по фотографии: free - одна фраза,
# structured - поля схемы Persona.
# Значение разбирает точка входа: пакет persona читает variables, и обратный
# импорт замкнул бы кольцо.
PERSONA_MODE = os.getenv("PERSONA_MODE", "free").strip().lower()

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



WEB_CONFIG = WebConfig(
    user_agent = WEB_USER_AGENT,
    request_timeout_seconds = WEB_TIMEOUT_SECONDS,
    max_page_bytes = WEB_MAX_PAGE_BYTES,
    cache_directory = _cache_directory(raw_path = WEB_CACHE_DIR),
    page_cache_ttl_days = WEB_PAGE_CACHE_TTL_DAYS,
    search_cache_ttl_days = WEB_SEARCH_CACHE_TTL_DAYS,
    bypass_cache = WEB_CACHE_BYPASS,
)

# Каталог журналов прогона. Пустое значение выключает журнал.
TRACE_DIR = _cache_directory(
    raw_path = os.getenv("TRACE_DIR", str(PROJECT_ROOT / "logs" / "traces")).strip()
)

# Каталог снимков состояния графа. Пустое значение выключает снимки, и тогда
# прогон нельзя переиграть с середины.
CHECKPOINT_DIR = _cache_directory(
    raw_path = os.getenv("CHECKPOINT_DIR", str(PROJECT_ROOT / ".cache" / "checkpoints")).strip()
)

# Отладочный режим: включить размышление на всех узлах, чем бы узел ни был
# занят. Нужен для разбора поведения модели. Прогон заметно дольше, поэтому по
# умолчанию выключен.
# он пришёл в ответе, - от этой переменной запись не зависит.
ENABLE_ALL_REASONING = os.getenv("ENABLE_ALL_REASONING", "").strip().lower() in ("1", "true", "yes")


# --- Речевой вход ---------------------------------------------------------
# Модель распознавания. На русском small путает имена собственные, medium уже
# держит, large-v3 на процессоре считает втрое дольше записи.
SPEECH_MODEL = os.getenv("SPEECH_MODEL", "medium")
SPEECH_DEVICE = os.getenv("SPEECH_DEVICE", "auto")
SPEECH_COMPUTE_TYPE = os.getenv("SPEECH_COMPUTE_TYPE", "int8")

# Язык записи. Пусто - определять по звуку, но на коротком вопросе whisper
# ошибается языком чаще, чем кажется.
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "ru").strip()

SPEECH_CACHE_DIR = os.getenv("SPEECH_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "speech")).strip()
SPEECH_CACHE_TTL_DAYS = int(os.getenv("SPEECH_CACHE_TTL_DAYS", "0"))
SPEECH_CACHE_BYPASS = os.getenv("SPEECH_CACHE_BYPASS", "").strip().lower() in ("1", "true", "yes")

# Whisper приводит звук к 16 кГц, писать выше незачем.
SPEECH_SAMPLE_RATE = 16_000

# Куда складывать записи с микрофона. Не временный каталог: записанный вопрос -
# материал, по нему прогон повторяется без микрофона.
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "recordings").strip()

LISTENING_CONFIG = ListeningConfig(
    recognition_model = SPEECH_MODEL,
    recognition_device = SPEECH_DEVICE,
    recognition_compute_type = SPEECH_COMPUTE_TYPE,
    language = SPEECH_LANGUAGE,
    cache_directory = _cache_directory(raw_path = SPEECH_CACHE_DIR),
    cache_ttl_days = SPEECH_CACHE_TTL_DAYS,
    bypass_cache = SPEECH_CACHE_BYPASS,
    recording_sample_rate = SPEECH_SAMPLE_RATE,
    recording_directory = _project_path(raw_path = RECORDINGS_DIR),
)
