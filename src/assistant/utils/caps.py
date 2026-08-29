"""
Нормализация регистра в тексте: снятие капслока.
"""

import re

# Доля слов, набранных заглавными, начиная с которой предложение считается капслоком.
CAPS_SHARE_THRESHOLD = 0.6

# Слово из букв, допускающее внутренние дефисы.
_WORD_PATTERN = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")

# Предложение: текст до знака конца предложения или до конца строки, вместе с отступом за ним.
_SENTENCE_PATTERN = re.compile(r"[^.!?…\n]*[.!?…]*\n?[ \t]*")

# Знаки, после которых начинается новое предложение.
_SENTENCE_END_MARKS = ".!?…"

# Знаки, которые стоят перед первым словом предложения и на его начало не влияют.
_OPENING_MARKS = "\"'«„([{-–—*#>"


def caps_word_share(text: str) -> float:
    """
    Считает долю слов, набранных заглавными, среди всех слов строки.

    Слово из одной буквы заглавным не считается: одиночная буква встречается
    в обозначениях и инициалах.

    Аргументы:
        text: строка для подсчёта.

    Возвращает:
        Долю от 0 до 1; для строки без слов - 0.
    """
    words = _WORD_PATTERN.findall(text)
    if not words:
        return 0.0
    caps_words = [word for word in words if word.isupper() and len(word) > 1]
    return len(caps_words) / len(words)


def _is_sentence_start(text: str, position: int) -> bool:
    """
    Проверяет, стоит ли слово в начале предложения.

    Аргументы:
        text: текст, в котором стоит слово.
        position: позиция первой буквы слова.

    Возвращает:
        True, если перед словом начало текста, перевод строки или конец предложения.
    """
    index = position - 1
    while index >= 0:
        character = text[index]
        if character == "\n":
            return True
        if character.isspace() or character in _OPENING_MARKS:
            index -= 1
            continue
        return character in _SENTENCE_END_MARKS
    return True


def build_case_reference(text: str) -> dict[str, str]:
    """
    Собирает образцы написания слов по тексту.

    В образцы попадает слово с заглавной буквы, стоящее не в начале предложения:
    такое слово считается именем собственным. Слова целиком заглавными
    пропускаются: в них написание не видно.

    Аргументы:
        text: текст-образец.

    Возвращает:
        Отображение слова в нижнем регистре на его написание из текста.
    """
    reference: dict[str, str] = {}
    for match in _WORD_PATTERN.finditer(text):
        word = match.group()
        if word.isupper() or not word[0].isupper():
            continue
        if _is_sentence_start(text = text, position = match.start()):
            continue
        reference.setdefault(word.lower(), word)
    return reference


def _normalize_sentence(sentence: str, case_reference: dict[str, str]) -> str:
    """
    Приводит предложение к обычному регистру.

    Слова из образцов сохраняют написание образца, остальные переводятся в нижний
    регистр, после чего восстанавливается заглавная в начале предложения.

    Аргументы:
        sentence: предложение целиком.
        case_reference: образцы написания слов из build_case_reference.

    Возвращает:
        Предложение в обычном регистре.
    """
    lowered = _WORD_PATTERN.sub(
        lambda match: case_reference.get(match.group().lower(), match.group().lower()),
        sentence,
    )

    characters = list(lowered)
    for match in _WORD_PATTERN.finditer(lowered):
        if _is_sentence_start(text = lowered, position = match.start()):
            characters[match.start()] = characters[match.start()].upper()
    return "".join(characters)


def normalize_caps(text: str, case_reference: dict[str, str], caps_threshold: float) -> str:
    """
    Приводит к обычному регистру те предложения текста, что набраны капслоком.

    Решение принимается по каждому предложению отдельно: предложение с долей слов
    заглавными ниже порога остаётся без изменений.

    Аргументы:
        text: исходный текст.
        case_reference: образцы написания слов из build_case_reference.
        caps_threshold: доля слов заглавными, начиная с которой предложение нормализуется.

    Возвращает:
        Текст, в котором капслок заменён обычным регистром.
    """
    normalized_sentences = []
    for match in _SENTENCE_PATTERN.finditer(text):
        sentence = match.group()
        if caps_word_share(text = sentence) < caps_threshold:
            normalized_sentences.append(sentence)
            continue
        normalized_sentences.append(
            _normalize_sentence(sentence = sentence, case_reference = case_reference)
        )
    return "".join(normalized_sentences)
