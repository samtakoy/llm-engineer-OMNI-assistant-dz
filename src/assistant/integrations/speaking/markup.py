"""
Санитайзер разметки ssml для silero.

sanitize_markup оставляет теги белого списка с допустимыми значениями,
остальную разметку превращает в текст, выбрасывает теги, разрезающие слово, и
закрывает незакрытые теги. Ударения знаком плюс и обычный текст проходят как
есть.

Белый список: prosody с rate и pitch, break с time. Абзацы и предложения
расставляет wrap_speech_parts по пустым строкам и знакам конца предложения:
это разбор текста, а не решение модели.
"""

import re
from xml.sax.saxutils import escape

from .voices import pitch_values, rate_values

# Тег целиком: закрывающая косая черта, имя, всё остальное до угловой скобки.
_TAG_PATTERN = re.compile(r"<\s*(/?)\s*([A-Za-z]+)([^>]*)>")

# Пара «имя атрибута - значение» в двойных либо одинарных кавычках.
_ATTRIBUTE_PATTERN = re.compile(r"([A-Za-z_-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")

# Длительность паузы: число с единицей.
_BREAK_TIME_PATTERN = re.compile(r"^\d{1,5}(ms|s)$")

# Тег с содержимым и парным закрывающим тегом.
_CONTAINER_TAGS = ("prosody",)

# Граница предложения: знак конца, за ним пробел.
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?…])\s+")

# Граница абзаца: пустая строка.
_PARAGRAPH_BOUNDARY_PATTERN = re.compile(r"\n\s*\n")

# Тег без содержимого.
_VOID_TAG = "break"

# Знак ударения считается частью слова наравне с буквами.
_ACCENT_MARK = "+"


def sanitize_markup(text: str) -> tuple[str, bool]:
    """
    Чистит разметку и отдаёт тело ssml.

    Аргументы:
        text: текст с разметкой или без неё.

    Возвращает:
        Пару «тело ssml, признак разметки». Признак истинен, если в теле остался
        хотя бы один тег белого списка.
    """
    tokens = _tokenize(text = text)
    _drop_stray_and_splitting_tags(tokens = tokens)

    return _render(tokens = tokens)


def wrap_speech_parts(body: str) -> str:
    """
    Расставляет теги абзацев и предложений в готовом теле ssml.

    Абзацы разделяет пустая строка, предложения - точка, восклицательный или
    вопросительный знак с пробелом. Границы внутри тегов не берутся.

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
    sentences: list[str] = []
    current: list[str] = []
    depth = 0
    position = 0

    for match in _TAG_PATTERN.finditer(paragraph):
        _append_text(
            sentences = sentences,
            current = current,
            chunk = paragraph[position:match.start()],
            is_splittable = depth == 0,
        )
        position = match.end()

        current.append(match.group(0))
        if match.group(2).lower() in _CONTAINER_TAGS:
            depth += -1 if match.group(1) == "/" else 1

    _append_text(
        sentences = sentences,
        current = current,
        chunk = paragraph[position:],
        is_splittable = depth == 0,
    )

    if current:
        sentences.append("".join(current))

    return "".join(f"<s>{sentence.strip()}</s>" for sentence in sentences if sentence.strip())


def _append_text(
    sentences: list[str],
    current: list[str],
    chunk: str,
    is_splittable: bool,
) -> None:
    """
    Добавляет кусок текста к текущему предложению, закрывая его на границах.

    Аргументы:
        sentences: собранные предложения; правится на месте.
        current: куски текущего предложения; правится на месте.
        chunk: кусок текста между тегами.
        is_splittable: делить кусок на предложения, иначе он идёт целиком.

    Возвращает:
        Ничего.
    """
    if not chunk:
        return

    if not is_splittable:
        current.append(chunk)
        return

    parts = _SENTENCE_BOUNDARY_PATTERN.split(chunk)
    for part in parts[:-1]:
        current.append(part)
        sentences.append("".join(current))
        current.clear()

    current.append(parts[-1])


def _tokenize(text: str) -> list[tuple[str, str]]:
    """
    Разбирает текст на куски текста и теги белого списка.

    Теги вне белого списка, prosody без допустимых атрибутов и пауза с плохой
    длительностью сюда не попадают.

    Аргументы:
        text: текст с разметкой или без неё.

    Возвращает:
        Список пар «вид куска, значение». Виды: text, open, close, void.
        У открывающего тега значение - готовая строка тега.
    """
    tokens: list[tuple[str, str]] = []
    position = 0

    for match in _TAG_PATTERN.finditer(text):
        chunk = text[position:match.start()]
        if chunk:
            tokens.append(("text", chunk))
        position = match.end()

        is_closing = match.group(1) == "/"
        name = match.group(2).lower()
        attributes = _parse_attributes(raw = match.group(3))

        if name == _VOID_TAG:
            time = attributes.get("time", "")
            if not is_closing and _BREAK_TIME_PATTERN.match(time):
                tokens.append(("void", f'<{_VOID_TAG} time="{time}"/>'))
            continue

        if name not in _CONTAINER_TAGS:
            continue

        if is_closing:
            tokens.append(("close", name))
            continue

        if match.group(3).strip().endswith("/"):
            continue

        opening = _render_opening_tag(name = name, attributes = attributes)
        if opening:
            tokens.append(("open", opening))

    chunk = text[position:]
    if chunk:
        tokens.append(("text", chunk))

    return tokens


def _tag_name(opening: str) -> str:
    """
    Достаёт имя тега из готовой строки открывающего тега.

    Аргументы:
        opening: строка вида «<prosody rate="fast">» или «<p>».

    Возвращает:
        Имя тега.
    """
    return opening.split()[0].strip("<>")


def _drop_stray_and_splitting_tags(tokens: list[tuple[str, str]]) -> None:
    """
    Выбрасывает закрывающие теги без пары и пары тегов, разрезающие слово.

    Аргументы:
        tokens: куски после разбора; правится на месте.

    Возвращает:
        Ничего.
    """
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []

    for index, (kind, value) in enumerate(tokens):
        if kind == "open":
            stack.append(index)
            continue
        if kind != "close":
            continue

        if stack and _tag_name(opening = tokens[stack[-1]][1]) == value:
            pairs.append((stack.pop(), index))
            continue

        tokens[index] = ("skip", "")

    for open_index, close_index in pairs:
        if _splits_word(tokens = tokens, open_index = open_index, close_index = close_index):
            tokens[open_index] = ("skip", "")
            tokens[close_index] = ("skip", "")


def _splits_word(tokens: list[tuple[str, str]], open_index: int, close_index: int) -> bool:
    """
    Проверяет, разрезает ли пара тегов слово.

    Аргументы:
        tokens: куски после разбора.
        open_index: место открывающего тега.
        close_index: место закрывающего тега.

    Возвращает:
        Истину, если открывающий или закрывающий тег стоит внутри слова.
    """
    before = _neighbour_character(tokens = tokens, index = open_index - 1, is_tail = True)
    after = _neighbour_character(tokens = tokens, index = close_index + 1, is_tail = False)

    inner = [value for kind, value in tokens[open_index + 1:close_index] if kind == "text"]
    inner_first = inner[0][0] if inner and inner[0] else ""
    inner_last = inner[-1][-1] if inner and inner[-1] else ""

    opens_inside_word = _is_word_character(character = before) and _is_word_character(character = inner_first)
    closes_inside_word = _is_word_character(character = inner_last) and _is_word_character(character = after)

    return opens_inside_word or closes_inside_word


def _neighbour_character(tokens: list[tuple[str, str]], index: int, is_tail: bool) -> str:
    """
    Отдаёт соседний символ, если соседний кусок - текст.

    Аргументы:
        tokens: куски после разбора.
        index: место соседнего куска.
        is_tail: брать последний символ куска, иначе первый.

    Возвращает:
        Символ либо пустую строку.
    """
    if index < 0 or index >= len(tokens):
        return ""

    kind, value = tokens[index]
    if kind != "text" or not value:
        return ""

    return value[-1] if is_tail else value[0]


def _is_word_character(character: str) -> bool:
    """
    Проверяет, что символ принадлежит слову.

    Аргументы:
        character: символ или пустая строка.

    Возвращает:
        Истину для букв, цифр и знака ударения.
    """
    return bool(character) and (character.isalnum() or character == _ACCENT_MARK)


def _render(tokens: list[tuple[str, str]]) -> tuple[str, bool]:
    """
    Собирает тело ssml из кусков.

    Аргументы:
        tokens: куски после чистки.

    Возвращает:
        Пару «тело ssml, признак разметки».
    """
    parts: list[str] = []
    opened: list[str] = []
    has_markup = False

    for kind, value in tokens:
        if kind == "text":
            parts.append(escape(value))
            continue
        if kind == "void":
            parts.append(value)
            has_markup = True
            continue
        if kind == "open":
            parts.append(value)
            opened.append(_tag_name(opening = value))
            has_markup = True
            continue
        if kind == "close" and opened:
            parts.append(f"</{opened.pop()}>")

    while opened:
        parts.append(f"</{opened.pop()}>")

    return "".join(parts), has_markup


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


def _render_opening_tag(name: str, attributes: dict[str, str]) -> str:
    """
    Собирает открывающий тег из допустимых атрибутов.

    Аргументы:
        name: имя тега из белого списка.
        attributes: атрибуты тега.

    Возвращает:
        Открывающий тег либо пустую строку, если тег отбрасывается.
    """
    if name != "prosody":
        return f"<{name}>"

    kept = []
    if attributes.get("rate") in rate_values():
        kept.append(f'rate="{attributes["rate"]}"')
    if attributes.get("pitch") in pitch_values():
        kept.append(f'pitch="{attributes["pitch"]}"')

    if not kept:
        return ""

    return f"<prosody {' '.join(kept)}>"
