"""
Санитайзер разметки ssml для silero.

sanitize_markup оставляет паузы белого списка, остальную разметку превращает в
текст, drop_markup выбрасывает разметку целиком, split_into_chunks режет длинный
текст на куски по бюджету символов. Ударения знаком плюс и обычный текст проходят
как есть.

Белый список: break с time. Темп и высоту голоса разметка не задаёт: они идут
одним значением на всю речь из настроек голоса. Абзацы и предложения
расставляет wrap_speech_parts по пустым строкам и знакам конца предложения:
это разбор текста, а не решение модели.
"""

import re
from xml.sax.saxutils import escape

# Тег целиком: закрывающая косая черта, имя, всё остальное до угловой скобки.
_TAG_PATTERN = re.compile(r"<\s*(/?)\s*([A-Za-z]+)([^>]*)>")

# Пара «имя атрибута - значение» в двойных либо одинарных кавычках.
_ATTRIBUTE_PATTERN = re.compile(r"([A-Za-z_-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")

# Длительность паузы: число с единицей.
_BREAK_TIME_PATTERN = re.compile(r"^\d{1,5}(ms|s)$")

# Граница предложения: знак конца, за ним пробел.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?…])\s+")

# Пробелы, оставшиеся на месте выброшенных тегов.
_SPACE_PATTERN = re.compile(r"[ \t]{2,}")

# Граница абзаца: пустая строка.
_PARAGRAPH_BOUNDARY_PATTERN = re.compile(r"\n\s*\n")

# Граница предложения вне тега: знак конца, пробел, и до ближайшей угловой скобки
# нет закрывающей - иначе рез пришёлся бы на середину тега.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+(?![^<>]*>)")

# Граница слова вне тега: запасной рез, когда предложение само длиннее бюджета.
_WORD_SPLIT_PATTERN = re.compile(r"\s+(?![^<>]*>)")

# Единственный тег белого списка; содержимого у него нет.
_VOID_TAG = "break"


def sanitize_markup(text: str) -> tuple[str, bool]:
    """
    Чистит разметку и отдаёт тело ssml.

    Аргументы:
        text: текст с разметкой или без неё.

    Возвращает:
        Пару «тело ssml, признак разметки». Признак истинен, если в теле
        осталась хотя бы одна пауза.
    """
    parts: list[str] = []
    has_markup = False
    position = 0

    for match in _TAG_PATTERN.finditer(text):
        chunk = text[position:match.start()]
        if chunk:
            parts.append(escape(chunk))
        position = match.end()

        pause = _rendered_pause(
            is_closing = match.group(1) == "/",
            name = match.group(2).lower(),
            raw_attributes = match.group(3),
        )
        if pause:
            parts.append(pause)
            has_markup = True

    chunk = text[position:]
    if chunk:
        parts.append(escape(chunk))

    return "".join(parts), has_markup


def drop_markup(text: str) -> str:
    """
    Выбрасывает разметку целиком и отдаёт текст для синтеза.

    Аргументы:
        text: текст с разметкой или без неё.

    Возвращает:
        Текст без единого тега.
    """
    return _SPACE_PATTERN.sub(" ", _TAG_PATTERN.sub("", text)).strip()


def split_into_chunks(text: str, budget: int) -> list[str]:
    """
    Режет текст на куски, каждый не длиннее бюджета символов.

    Рез идёт по концам предложений, предложение длиннее бюджета дорезается по
    пробелам. Границы внутри тегов не берутся: разрезанный тег ушёл бы в речь
    мусором. Длина куска считается по тексту без разметки: столько же символов
    считает модель.

    Аргументы:
        text: текст с разметкой или без неё.
        budget: сколько символов без разметки помещается в один кусок.

    Возвращает:
        Куски в порядке чтения. Слово длиннее бюджета остаётся куском длиннее
        бюджета: резать дальше нечего.
    """
    chunks: list[str] = []

    for sentence in _SENTENCE_SPLIT_PATTERN.split(text):
        if len(drop_markup(text = sentence)) <= budget:
            parts = [sentence]
        else:
            parts = _WORD_SPLIT_PATTERN.split(sentence)

        for part in parts:
            merged = f"{chunks[-1]} {part}" if chunks else part
            if chunks and len(drop_markup(text = merged)) <= budget:
                chunks[-1] = merged
            else:
                chunks.append(part)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def wrap_speech_parts(body: str) -> str:
    """
    Расставляет теги абзацев и предложений в готовом теле ssml.

    Абзацы разделяет пустая строка, предложения - точка, восклицательный или
    вопросительный знак с пробелом.

    Аргументы:
        body: тело ssml после санитайзера.

    Возвращает:
        Тело, разложенное по тегам p и s.
    """
    paragraphs = [
        paragraph
        for paragraph in _PARAGRAPH_BOUNDARY_PATTERN.split(body)
        if paragraph.strip()
    ]

    return "".join(f"<p>{_wrap_sentences(paragraph = paragraph)}</p>" for paragraph in paragraphs)


def _wrap_sentences(paragraph: str) -> str:
    """
    Разбивает абзац на предложения и оборачивает каждое тегом s.

    Аргументы:
        paragraph: часть тела ssml без пустых строк.

    Возвращает:
        Предложения абзаца в тегах s.
    """
    sentences = _SENTENCE_BOUNDARY_PATTERN.split(paragraph)

    return "".join(f"<s>{sentence.strip()}</s>" for sentence in sentences if sentence.strip())


def _rendered_pause(is_closing: bool, name: str, raw_attributes: str) -> str:
    """
    Собирает тег паузы из разобранного тега.

    Аргументы:
        is_closing: тег закрывающий.
        name: имя тега в нижнем регистре.
        raw_attributes: часть тега после имени.

    Возвращает:
        Тег паузы либо пустую строку, если тег не пауза или длительность плохая.
    """
    if is_closing or name != _VOID_TAG:
        return ""

    time = _parse_attributes(raw = raw_attributes).get("time", "")
    if not _BREAK_TIME_PATTERN.match(time):
        return ""

    return f'<{_VOID_TAG} time="{time}"/>'


def _parse_attributes(raw: str) -> dict[str, str]:
    """
    Разбирает атрибуты тега.

    Аргументы:
        raw: часть тега после имени.

    Возвращает:
        Значения по именам атрибутов в нижнем регистре.
    """
    attributes: dict[str, str] = {}
    for match in _ATTRIBUTE_PATTERN.finditer(raw):
        value = match.group(2) if match.group(2) is not None else match.group(3)
        attributes[match.group(1).lower()] = value.strip()

    return attributes
