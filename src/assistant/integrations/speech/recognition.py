"""
Распознавание речи поверх faster-whisper, с кешем расшифровок.

Модель живёт в поле объекта, а не в переменной модуля: она весит от сотен
мегабайт до полутора гигабайт, грузится секундами и должна переживать несколько
вызовов. Кто создал распознаватель, тот и решает, сколько он живёт.

Расшифровка кладётся в кеш: whisper считает запись десятки секунд, а сама
запись не меняется. Ключ - отпечаток содержимого файла вместе с моделью и
языком: смена модели даёт другую запись, и старая расшифровка не выдаётся за
новую.

Наружу исключения не уходят: не нашёлся файл, не установлена библиотека, не
скачалась модель - всё это возвращается причиной в исходе.
"""

import hashlib
from pathlib import Path
from typing import Any

from ..filecache import open_cache
from .config import SpeechConfig
from .outcomes import TranscriptOutcome

# Версия формата записи о расшифровке.
_RECORD_VERSION = 1

_CACHE_NAMESPACE = "transcripts"

# Сколько байт файла читать за раз при подсчёте отпечатка.
_FINGERPRINT_CHUNK_BYTES = 1_000_000


class SpeechRecognizer:
    """
    Распознаватель речи с отложенной загрузкой модели.

    Модель грузится при первом распознавании: создание объекта ничего не стоит,
    и проект, который в этот прогон говорить не собирался, не платит за неё.
    """

    def __init__(self, config: SpeechConfig) -> None:
        """
        Аргументы:
            config: настройки речевого слоя.
        """
        self._config = config
        self._model: Any | None = None
        self._cache = open_cache(
            directory = config.cache_directory,
            namespace = _CACHE_NAMESPACE,
            ttl_days = config.cache_ttl_days,
            record_version = _RECORD_VERSION,
        )

    def transcribe(self, audio_path: Path) -> TranscriptOutcome:
        """
        Превращает запись в текст.

        Аргументы:
            audio_path: файл с записью. Формат любой, faster-whisper декодирует
                и приводит звук к нужной частоте сам.

        Возвращает:
            Исход распознавания: текст либо причина неудачи.
        """
        if not audio_path.is_file():
            return TranscriptOutcome(
                text = "",
                error = f"файла {audio_path} нет",
                from_cache = False,
            )

        key = self._cache_key(audio_path = audio_path)
        if key and self._cache is not None and not self._config.bypass_cache:
            record = self._cache.read(key = key)
            if record is not None and record.get("text"):
                return TranscriptOutcome(text = record["text"], error = "", from_cache = True)

        model, load_error = self._loaded_model()
        if model is None:
            return TranscriptOutcome(text = "", error = load_error, from_cache = False)

        try:
            segments, info = model.transcribe(
                str(audio_path),
                language = self._config.language or None,
                vad_filter = True,
            )
            # Сегменты приезжают ленивым генератором: распознавание идёт здесь,
            # а не в вызове transcribe.
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as error:
            print(f"[speech] распознавание {audio_path.name} не удалось: {type(error).__name__}: {error}")
            return TranscriptOutcome(
                text = "",
                error = f"распознавание оборвалось: {type(error).__name__}",
                from_cache = False,
            )

        if not text:
            return TranscriptOutcome(
                text = "",
                error = "в записи не нашлось речи",
                from_cache = False,
            )

        if key and self._cache is not None:
            self._cache.write(
                key = key,
                payload = {
                    "audio_name": audio_path.name,
                    "model": self._config.recognition_model,
                    "language": info.language,
                    "duration_seconds": round(info.duration, 2),
                    "text": text,
                },
            )

        return TranscriptOutcome(text = text, error = "", from_cache = False)

    def _loaded_model(self) -> tuple[Any | None, str]:
        """
        Отдаёт загруженную модель, загружая её при первом обращении.

        Возвращает:
            Пару «модель, причина неудачи». При успехе причина пустая, при
            неудаче модель None.
        """
        if self._model is not None:
            return self._model, ""

        try:
            from faster_whisper import WhisperModel
        except (ImportError, OSError) as error:
            # OSError ловится наравне с ImportError: ctranslate2 подгружает свои
            # бинарники на импорте и без них падает именно так.
            print(f"[speech] библиотека faster-whisper недоступна: {type(error).__name__}: {error}")
            return None, "библиотека распознавания недоступна"

        print(
            f"[speech] загрузка модели {self._config.recognition_model} "
            f"({self._config.recognition_device}, {self._config.recognition_compute_type})"
        )
        try:
            self._model = WhisperModel(
                self._config.recognition_model,
                device = self._config.recognition_device,
                compute_type = self._config.recognition_compute_type,
            )
        except Exception as error:
            print(f"[speech] модель не загрузилась: {type(error).__name__}: {error}")
            return None, f"модель {self._config.recognition_model} не загрузилась"

        return self._model, ""

    def _cache_key(self, audio_path: Path) -> str:
        """
        Составляет ключ записи о расшифровке.

        Отпечаток берётся по содержимому файла, а не по имени: одну и ту же
        запись можно переложить и переименовать, а переслушивать её незачем.

        Аргументы:
            audio_path: файл с записью.

        Возвращает:
            Ключ записи либо пустую строку, если файл не прочитался и кеш
            в этот раз не работает.
        """
        digest = hashlib.sha1()
        try:
            with audio_path.open("rb") as stream:
                while chunk := stream.read(_FINGERPRINT_CHUNK_BYTES):
                    digest.update(chunk)
        except OSError as error:
            print(f"[speech] отпечаток {audio_path.name} не снялся: {type(error).__name__}: {error}")
            return ""

        return (
            f"{digest.hexdigest()}|{self._config.recognition_model}|{self._config.language or 'auto'}"
        )
