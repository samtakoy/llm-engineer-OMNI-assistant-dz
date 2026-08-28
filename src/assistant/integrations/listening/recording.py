"""
Запись вопроса с микрофона в wav.

Два режима: фиксированная длительность и запись до нажатия Enter. Первый нужен,
когда запуск идёт из скрипта, второй - когда человек не знает заранее, сколько
будет говорить.

Файл кладётся не во временный каталог: записанный вопрос - материал. Путь
печатается, и дальше тот же вопрос гоняется по файлу сколько угодно раз, без
микрофона и без повторного распознавания.

Наружу исключения не уходят: нет библиотеки, нет разрешения на микрофон, нет
устройства ввода - всё это возвращается причиной в исходе.
"""

import wave
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import ListeningConfig
from .outcomes import RecordingOutcome

# Ширина отсчёта: пишем в int16, так же его понимает wave и любой декодер.
_SAMPLE_WIDTH_BYTES = 2
_MAX_AMPLITUDE = 32768.0

# Ниже этого уровня запись считается тишиной: микрофон не тот, не выдано
# разрешение или человек не говорил.
SILENCE_LEVEL = 0.01


def record(seconds: float | None, config: ListeningConfig) -> RecordingOutcome:
    """
    Пишет звук с микрофона в файл.

    Аргументы:
        seconds: сколько секунд писать. None - писать до нажатия Enter.
        config: настройки речевого слоя.

    Возвращает:
        Исход записи: путь к файлу либо причина неудачи.
    """
    try:
        import numpy
        import sounddevice
    except (ImportError, OSError) as error:
        # OSError, а не только ImportError: колёса sounddevice для linux идут без
        # portaudio, и на импорте он ищет системную библиотеку. Нет её - импорт
        # падает OSError, и без этой ветки запись роняла бы прогон.
        print(f"[speech] запись недоступна: {type(error).__name__}: {error}")
        return RecordingOutcome(
            path = None,
            error = "библиотека записи недоступна: нет sounddevice или portaudio",
            seconds = 0.0,
            peak_level = 0.0,
        )

    try:
        if seconds is None:
            samples = _record_until_enter(
                sounddevice = sounddevice,
                numpy = numpy,
                sample_rate = config.recording_sample_rate,
            )
        else:
            samples = _record_fixed(
                sounddevice = sounddevice,
                seconds = seconds,
                sample_rate = config.recording_sample_rate,
            )
    except Exception as error:
        print(f"[speech] запись не удалась: {type(error).__name__}: {error}")
        return RecordingOutcome(
            path = None,
            error = f"микрофон недоступен: {type(error).__name__}",
            seconds = 0.0,
            peak_level = 0.0,
        )

    if len(samples) == 0:
        return RecordingOutcome(
            path = None,
            error = "записывать оказалось нечего",
            seconds = 0.0,
            peak_level = 0.0,
        )

    peak_level = float(numpy.max(numpy.abs(samples))) / _MAX_AMPLITUDE
    duration_seconds = len(samples) / config.recording_sample_rate

    path = _recording_path(directory = config.recording_directory)
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(_SAMPLE_WIDTH_BYTES)
            target.setframerate(config.recording_sample_rate)
            target.writeframes(samples.tobytes())
    except Exception as error:
        print(f"[speech] запись не сохранилась: {type(error).__name__}: {error}")
        return RecordingOutcome(
            path = None,
            error = "файл записи не сохранился",
            seconds = duration_seconds,
            peak_level = peak_level,
        )

    return RecordingOutcome(
        path = path,
        error = "",
        seconds = duration_seconds,
        peak_level = peak_level,
    )


def _record_fixed(sounddevice: ModuleType, seconds: float, sample_rate: int) -> Any:
    """
    Пишет заданное число секунд.

    Аргументы:
        sounddevice: модуль записи.
        seconds: длительность записи.
        sample_rate: частота дискретизации.

    Возвращает:
        Массив отсчётов int16 одним каналом.
    """
    print(f"[speech] запись {seconds:.0f} секунд, говорите")
    samples = sounddevice.rec(
        int(seconds * sample_rate),
        samplerate = sample_rate,
        channels = 1,
        dtype = "int16",
    )
    sounddevice.wait()
    return samples.reshape(-1)


def _record_until_enter(sounddevice: ModuleType, numpy: ModuleType, sample_rate: int) -> Any:
    """
    Пишет, пока не нажат Enter.

    Аргументы:
        sounddevice: модуль записи.
        numpy: модуль массивов.
        sample_rate: частота дискретизации.

    Возвращает:
        Массив отсчётов int16 одним каналом.
    """
    chunks = []

    def collect(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Складывает очередной кусок звука. Вызывается из потока записи."""
        if status:
            print(f"[speech] поток записи: {status}")
        chunks.append(indata.copy())

    print("[speech] запись пошла, говорите. Enter - остановить")
    with sounddevice.InputStream(
        samplerate = sample_rate,
        channels = 1,
        dtype = "int16",
        callback = collect,
    ):
        input()

    if not chunks:
        return numpy.empty(0, dtype = "int16")

    return numpy.concatenate(chunks).reshape(-1)


def _recording_path(directory: Path) -> Path:
    """
    Составляет путь к новому файлу записи.

    Имя - момент записи: так файлы сортируются по времени и не затирают друг
    друга.

    Аргументы:
        directory: каталог для записей.

    Возвращает:
        Путь к файлу.
    """
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return directory / f"{stamp}.wav"
