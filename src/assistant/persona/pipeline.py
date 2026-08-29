"""
Разбор фотографии персонажа: облик текстом, кеш разборов и сборка рассказчика.

Наружу исключения не уходят - причина возвращается второй половиной пары.
"""

import hashlib
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..integrations.filecache import open_cache
from ..integrations.llm.vision import describe_image, image_data_url
from ..variables import (
    VISION_CACHE_BYPASS,
    VISION_CACHE_DIR,
    VISION_CACHE_TTL_DAYS,
    VISION_JPEG_QUALITY,
    VISION_MAX_SIDE,
)
from .prompts import LOOK_PROMPT, NARRATOR_STYLE_PROMPT, PERSONA_PROMPT
from .schemas import Persona

# Подкаталог кеша и версия формата записи. Версия поднимается при смене
# состава полей записи.
_CACHE_NAMESPACE = "vision"
_RECORD_VERSION = 1

# Отпечаток файла снимается кусками: фотография целиком в память не нужна.
_FINGERPRINT_CHUNK_BYTES = 1024 * 1024

# Сколько знаков отпечатка промпта класть в ключ.
_PROMPT_FINGERPRINT_LENGTH = 8


def _cache_key(image_path: Path, model_name: str, prompt: str) -> str:
    """
    Составляет ключ записи о разборе картинки.

    Отпечаток берётся по содержимому файла, а не по имени: одну и ту же
    фотографию можно переложить и переименовать, а пересматривать её незачем.
    Имя модели и отпечаток промпта входят в ключ, потому что от них зависит
    ответ.

    Аргументы:
        image_path: файл с картинкой.
        model_name: имя модели, которая смотрит картинку.
        prompt: текст инструкции для модели.

    Возвращает:
        Ключ записи либо пустую строку, если файл не прочитался и кеш в этот
        раз не работает.
    """
    digest = hashlib.sha1()
    try:
        with image_path.open("rb") as stream:
            while chunk := stream.read(_FINGERPRINT_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        print(f"[персона] отпечаток {image_path.name} не снялся: {type(error).__name__}: {error}")
        return ""

    prompt_fingerprint = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:_PROMPT_FINGERPRINT_LENGTH]

    return f"{digest.hexdigest()}|{model_name}|{prompt_fingerprint}"


def describe_look(llm: ChatOpenAI, image_path: Path) -> tuple[str, str]:
    """
    Разбирает фотографию персонажа и возвращает облик текстом.

    Ответ приходит свободным текстом, а не структурой.

    Аргументы:
        llm: клиент модели, принимающей картинки.
        image_path: файл с фотографией персонажа.

    Возвращает:
        Пару «описание облика, причина неудачи». При успехе причина пустая, при
        неудаче описание пустое.
    """
    cache = open_cache(
        directory = VISION_CACHE_DIR,
        namespace = _CACHE_NAMESPACE,
        ttl_days = VISION_CACHE_TTL_DAYS,
        record_version = _RECORD_VERSION,
    )

    key = _cache_key(image_path = image_path, model_name = llm.model_name, prompt = LOOK_PROMPT)
    if key and cache is not None and not VISION_CACHE_BYPASS:
        record = cache.read(key = key)
        if record is not None and record.get("look"):
            return record["look"], ""

    image_url, error = image_data_url(
        image_path = image_path,
        max_side = VISION_MAX_SIDE,
        jpeg_quality = VISION_JPEG_QUALITY,
    )
    if error:
        return "", error

    look, error = describe_image(
        llm = llm,
        image_url = image_url,
        instruction = LOOK_PROMPT,
    )
    if error:
        return "", error

    if key and cache is not None:
        cache.write(key = key, payload = {"look": look, "image": image_path.name})

    return look, ""


def build_persona(llm: ChatOpenAI, look: str) -> tuple[Persona | None, str]:
    """
    Придумывает рассказчика по описанию облика.

    Аргументы:
        llm: клиент текстовой модели.
        look: описание облика персонажа.

    Возвращает:
        Пару «персонаж, причина неудачи». При успехе причина пустая, при
        неудаче персонаж None.
    """
    if not look.strip():
        return None, "описание облика пустое"

    structured_llm = llm.with_structured_output(Persona, method = "json_schema")

    try:
        persona = structured_llm.invoke(
            [
                SystemMessage(content = PERSONA_PROMPT),
                HumanMessage(content = f"Описание облика:\n{look}"),
            ]
        )
    except Exception as error:
        print(f"[персона] вызов модели не удался: {type(error).__name__}: {error}")
        return None, f"модель не ответила по схеме: {type(error).__name__}"

    return persona, ""


def build_narrator_style(llm: ChatOpenAI, look: str) -> tuple[str, str]:
    """
    Сжимает описание облика до одной фразы про голос рассказчика.

    Аргументы:
        llm: клиент текстовой модели.
        look: описание облика персонажа.

    Возвращает:
        Пару «фраза, причина неудачи». При успехе причина пустая, при неудаче
        фраза пустая.
    """
    if not look.strip():
        return "", "описание облика пустое"

    try:
        message = llm.invoke(
            [
                SystemMessage(content = NARRATOR_STYLE_PROMPT),
                HumanMessage(content = f"Описание облика:\n{look}"),
            ]
        )
    except Exception as error:
        print(f"[персона] вызов модели не удался: {type(error).__name__}: {error}")
        return "", f"модель не ответила: {type(error).__name__}"

    style = message.text.strip()
    if not style:
        return "", "модель вернула пустой ответ"

    return style, ""
