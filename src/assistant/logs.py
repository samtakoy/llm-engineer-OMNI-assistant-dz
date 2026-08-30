"""
Ход прогона вне графа в журнал: вопрос, облик, рассказчик, голос, разметка,
озвучка.

Строки идут в logging под именем модуля; куда их показать, решает настройка
вывода в пакете observability. Журнал прогона слушает тот же логгер, поэтому
этапы вокруг графа попадают в один файл с графом.
"""

import logging
from pathlib import Path

from assistant.integrations.speaking import VoiceSettings
from assistant.persona import Persona
from assistant.timing import Stopwatch

logger = logging.getLogger(__name__)


def log_recording(path: Path, seconds: float, peak_level: float) -> None:
    """
    Пишет готовую запись с микрофона.

    Аргументы:
        path: файл с записью.
        seconds: длительность записи.
        peak_level: громкость самого громкого отсчёта, от нуля до единицы.

    Возвращает:
        Ничего.
    """
    logger.info(f"[запись] {path} - {seconds:.1f} с, громкость {peak_level:.2f}")


def log_recording_silent() -> None:
    """
    Пишет, что в записи тишина, и что проверить.

    Возвращает:
        Ничего.
    """
    logger.info(
        "[запись] в файле тишина. На macOS проверьте разрешение на микрофон "
        "у терминала в настройках приватности и выбранное устройство ввода."
    )


def log_question(question: str, is_from_cache: bool) -> None:
    """
    Пишет распознанный вопрос.

    Аргументы:
        question: текст вопроса.
        is_from_cache: расшифровка взята из кеша, а не посчитана заново.

    Возвращает:
        Ничего.
    """
    source = "из кеша" if is_from_cache else "распознано"
    logger.info(f"[вопрос] {source}: {question}")


def log_look(look: str) -> None:
    """
    Пишет описание облика с фотографии.

    Аргументы:
        look: описание облика.

    Возвращает:
        Ничего.
    """
    logger.info(f"[облик]\n{look.strip()}")


def log_persona(persona: Persona) -> None:
    """
    Пишет поля рассказчика.

    Аргументы:
        persona: рассказчик, выведенный из облика.

    Возвращает:
        Ничего.
    """
    logger.info(
        f"[рассказчик] {persona.name} ({persona.gender})\n"
        f"характер: {persona.character}\n"
        f"обращение: {persona.address_to_listener}\n"
        f"манера: {persona.speech_manner}\n"
        f"словечки: {', '.join(persona.favourite_words)}\n"
        f"звуки: {', '.join(persona.favourite_sounds)}\n"
        f"отношение к предмету: {persona.attitude_to_subject}"
    )


def log_narrator_style(style: str) -> None:
    """
    Пишет фразу о голосе рассказчика.

    Аргументы:
        style: фраза, задающая манеру изложения.

    Возвращает:
        Ничего.
    """
    logger.info(f"[рассказчик]\n{style.strip()}")


def log_voice(settings: VoiceSettings) -> None:
    """
    Пишет выбранные настройки голоса.

    Аргументы:
        settings: голос, темп, высота и эффект.

    Возвращает:
        Ничего.
    """
    logger.info(
        f"[голос] {settings.speaker}, темп {settings.rate}, высота {settings.pitch}, "
        f"эффект {settings.effect} ({settings.effect_strength})"
    )


def log_title_voice(speaker: str) -> None:
    """
    Пишет голос диктора, читающего заголовки разделов.

    Аргументы:
        speaker: имя голоса диктора.

    Возвращает:
        Ничего.
    """
    logger.info(f"[голос] заголовки читает {speaker}")


def log_title_voice_missing(speaker: str) -> None:
    """
    Пишет, что голоса диктора нет в модели и заголовки читает голос персонажа.

    Аргументы:
        speaker: имя голоса, которого не нашлось.

    Возвращает:
        Ничего.
    """
    logger.info(f"[голос] голоса диктора {speaker} нет в модели, заголовки читает персонаж")


def log_voices_missing(reason: str) -> None:
    """
    Пишет, что список голосов не получен и озвучки не будет.

    Аргументы:
        reason: причина неудачи.

    Возвращает:
        Ничего.
    """
    logger.info(f"[голос] голоса не получены: {reason}")


def log_voice_fallback(reason: str) -> None:
    """
    Пишет, что голос не подобран и берётся голос по умолчанию.

    Аргументы:
        reason: причина неудачи подбора.

    Возвращает:
        Ничего.
    """
    logger.info(f"[голос] голос не подобран: {reason}")


def log_markup(index: int, marked_text: str) -> None:
    """
    Пишет разметку куска, вернувшуюся от модели.

    Аргументы:
        index: номер куска, считая с единицы.
        marked_text: текст с разметкой.

    Возвращает:
        Ничего.
    """
    logger.info(f"[разметка] кусок {index}\n{marked_text.strip()}")


def log_markup_skipped(index: int, reason: str) -> None:
    """
    Пишет, что кусок озвучивается без разметки.

    Аргументы:
        index: номер куска, считая с единицы.
        reason: причина неудачи разметки.

    Возвращает:
        Ничего.
    """
    logger.info(f"[разметка] кусок {index} без разметки: {reason}")


def log_spoken(index: int, path: Path, seconds: float) -> None:
    """
    Пишет готовый файл с озвучкой куска.

    Аргументы:
        index: номер куска, считая с единицы.
        path: файл со звуком.
        seconds: длительность звучания.

    Возвращает:
        Ничего.
    """
    logger.info(f"[озвучка] кусок {index}: {path}, звучание {seconds:.1f} с")


def log_speech_failed(index: int, reason: str) -> None:
    """
    Пишет, что кусок озвучить не вышло.

    Аргументы:
        index: номер куска, считая с единицы.
        reason: причина неудачи.

    Возвращает:
        Ничего.
    """
    logger.info(f"[озвучка] кусок {index} не озвучен: {reason}")


def log_timing(timing: Stopwatch) -> None:
    """
    Пишет таблицу длительностей этапов.

    Аргументы:
        timing: копилка замеров.

    Возвращает:
        Ничего.
    """
    table = timing.render_table()
    if table:
        logger.info(f"[длительности]\n{table}")
