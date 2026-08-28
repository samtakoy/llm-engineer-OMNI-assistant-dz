"""
Служебный слой зрения: картинка в тело запроса и вызов модели по схеме.

Наружу исключения не уходят - возвращается причиной второй половиной пары.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel

# Схема ответа приходит от вызывающего, и вернуть надо её же, а не BaseModel:
# иначе у поля разобранного ответа теряется тип.
StructuredAnswer = TypeVar("StructuredAnswer", bound = BaseModel)

# Формат пережатия. Один на все входные картинки: блок image_url принимает
# jpeg от любого сервера, а разбор входного формата не стоит своей ветки кода.
_ENCODED_FORMAT = "JPEG"
_ENCODED_MIME_TYPE = "image/jpeg"


def image_data_url(image_path: Path, max_side: int, jpeg_quality: int) -> tuple[str, str]:
    """
    Приводит картинку к строке data url для блока image_url.

    Большая сторона ужимается до max_side.

    Аргументы:
        image_path: файл с картинкой.
        max_side: предел большей стороны в пикселях.
        jpeg_quality: качество пережатия, от 1 до 95.

    Возвращает:
        Пару «строка data url, причина неудачи». При успехе причина пустая,
        при неудаче строка пустая.
    """
    if not image_path.is_file():
        return "", f"файла {image_path} нет"

    buffer = BytesIO()
    try:
        with Image.open(image_path) as picture:
            # Приведение к rgb обязательно: jpeg не принимает ни прозрачность,
            # ни палитру, а на входе бывает и png, и gif.
            frame = picture.convert("RGB")
            frame.thumbnail((max_side, max_side))
            frame.save(buffer, format = _ENCODED_FORMAT, quality = jpeg_quality)
    except Exception as error:
        print(f"[зрение] картинка {image_path.name} не открылась: {type(error).__name__}: {error}")
        return "", f"картинка {image_path.name} не открылась"

    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{_ENCODED_MIME_TYPE};base64,{payload}", ""


def _image_message(image_url: str, instruction: str) -> HumanMessage:
    """
    Складывает сообщение с текстом и картинкой.

    Аргументы:
        image_url: картинка строкой data url либо обычным адресом.
        instruction: что спросить у картинки.

    Возвращает:
        Сообщение для отправки модели.
    """
    return HumanMessage(
        content = [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )


def describe_image(llm: ChatOpenAI, image_url: str, instruction: str) -> tuple[str, str]:
    """
    Спрашивает модель о картинке и возвращает ответ текстом.

    Аргументы:
        llm: клиент модели, собранный build_llm с моделью, принимающей картинки.
        image_url: картинка строкой data url либо обычным адресом.
        instruction: что спросить у картинки.

    Возвращает:
        Пару «текст ответа, причина неудачи». При успехе причина пустая, при
        неудаче текст пустой.
    """
    try:
        message = llm.invoke([_image_message(image_url = image_url, instruction = instruction)])
    except Exception as error:
        print(f"[зрение] вызов модели не удался: {type(error).__name__}: {error}")
        return "", f"модель не ответила: {type(error).__name__}"

    # Обрыв по потолку длины не ошибка, ответ приходит урезанным. Без пометки
    # его не отличить от короткого описания.
    if message.response_metadata.get("finish_reason") == "length":
        print("[зрение] ответ обрезан потолком длины")

    text = message.text.strip()
    if not text:
        return "", "модель вернула пустой ответ"

    return text, ""


def look_at_image(
    llm: ChatOpenAI,
    image_url: str,
    instruction: str,
    schema: type[StructuredAnswer],
) -> tuple[StructuredAnswer | None, str]:
    """
    Спрашивает модель о картинке и разбирает ответ по схеме.

    Путь для моделей, которые держат грамматику. Мелкой vl-модели подходит
    соседняя describe_image.

    Аргументы:
        llm: клиент модели, собранный build_llm с моделью, принимающей картинки.
        image_url: картинка строкой data url либо обычным адресом.
        instruction: что спросить у картинки.
        schema: схема ответа.

    Возвращает:
        Пару «разобранный ответ, причина неудачи». При успехе причина пустая,
        при неудаче ответ None.
    """
    structured_llm = llm.with_structured_output(schema, method = "json_schema")

    try:
        answer = structured_llm.invoke(
            [_image_message(image_url = image_url, instruction = instruction)]
        )
    except Exception as error:
        print(f"[зрение] вызов модели не удался: {type(error).__name__}: {error}")
        return None, f"модель не ответила по схеме: {type(error).__name__}"

    return answer, ""
