"""
Синтез речи моделью silero и запись звука в wav.

SpeechSynthesizer отдаёт список голосов модели и озвучивает текст заданным
голосом, темпом, высотой и звуковым эффектом. Модель грузится при первой
озвучке и остаётся в поле объекта.

Ненейтральные темп и высота уходят в модель разметкой ssml, нейтральные - чистым
текстом с ударениями и буквой ё: в режиме ssml silero флаги ударений не
принимает.

При неудаче возвращается исход с причиной, исключения наружу не уходят.
"""

import wave
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .config import SpeakingConfig
from .effects import apply_effect
from .outcomes import SynthesisOutcome
from .voices import VoiceSettings

# Откуда torch.hub берёт код silero и файл весов.
_HUB_REPOSITORY = "snakers4/silero-models"
_HUB_MODEL = "silero_tts"

# Ширина отсчёта: пишем в int16, так же его понимает wave и любой декодер.
_SAMPLE_WIDTH_BYTES = 2
_MAX_AMPLITUDE = 32767

# Темп и высота, при которых разметка не нужна.
_NEUTRAL_PROSODY = "medium"


class SpeechSynthesizer:
    """
    Синтезатор речи: список голосов модели и озвучка текста в wav.

    Модель грузится при первом обращении и остаётся в поле объекта.
    """

    def __init__(self, config: SpeakingConfig) -> None:
        """
        Аргументы:
            config: настройки синтеза.
        """
        self._config = config
        self._model: Any | None = None

    def available_speakers(self) -> tuple[list[str], str]:
        """
        Отдаёт голоса, которые знает загруженная версия модели.

        Возвращает:
            Пару «список имён голосов, причина неудачи». При неудаче список
            пустой.
        """
        model, load_error = self._loaded_model()
        if model is None:
            return [], load_error

        return list(model.speakers), ""

    def synthesize(
        self,
        text: str,
        settings: VoiceSettings,
        output_path: Path,
    ) -> SynthesisOutcome:
        """
        Озвучивает текст заданным голосом, накладывает эффект и кладёт звук в файл.

        Аргументы:
            text: что произнести.
            settings: голос, темп, высота и звуковой эффект.
            output_path: файл, куда писать звук.

        Возвращает:
            Исход озвучки: путь к файлу либо причина неудачи.
        """
        spoken_text = text.strip()
        if not spoken_text:
            return SynthesisOutcome(path = None, error = "озвучивать нечего", seconds = 0.0)

        model, load_error = self._loaded_model()
        if model is None:
            return SynthesisOutcome(path = None, error = load_error, seconds = 0.0)

        speakers = list(model.speakers)
        if settings.speaker not in speakers:
            return SynthesisOutcome(
                path = None,
                error = f"голоса {settings.speaker} нет в модели {self._config.model_id}",
                seconds = 0.0,
            )

        try:
            audio = self._apply_tts(model = model, text = spoken_text, settings = settings)
        except Exception as error:
            print(f"[speaking] озвучка не удалась: {type(error).__name__}: {error}")
            return SynthesisOutcome(
                path = None,
                error = f"озвучка оборвалась: {type(error).__name__}",
                seconds = 0.0,
            )

        audio, effect_error = apply_effect(
            audio = audio,
            sample_rate = self._config.sample_rate,
            effect = settings.effect,
            strength = settings.effect_strength,
        )
        if effect_error:
            return SynthesisOutcome(path = None, error = effect_error, seconds = 0.0)

        seconds = len(audio) / self._config.sample_rate
        write_error = self._write_wav(audio = audio, output_path = output_path)
        if write_error:
            return SynthesisOutcome(path = None, error = write_error, seconds = seconds)

        return SynthesisOutcome(path = output_path, error = "", seconds = seconds)

    def _apply_tts(self, model: Any, text: str, settings: VoiceSettings) -> Any:
        """
        Зовёт синтез: чистым текстом либо разметкой ssml.

        Аргументы:
            model: загруженная модель silero.
            text: что произнести.
            settings: голос, темп и высота.

        Возвращает:
            Отсчёты звука тензором в диапазоне от минус единицы до единицы.
        """
        is_neutral = settings.rate == _NEUTRAL_PROSODY and settings.pitch == _NEUTRAL_PROSODY
        if is_neutral:
            return model.apply_tts(
                text = text,
                speaker = settings.speaker,
                sample_rate = self._config.sample_rate,
                put_accent = self._config.put_accent,
                put_yo = self._config.put_yo,
            )

        return model.apply_tts(
            ssml_text = _render_ssml(text = text, settings = settings),
            speaker = settings.speaker,
            sample_rate = self._config.sample_rate,
        )

    def _write_wav(self, audio: Any, output_path: Path) -> str:
        """
        Пишет отсчёты в wav.

        Аргументы:
            audio: отсчёты звука тензором от минус единицы до единицы.
            output_path: файл, куда писать звук.

        Возвращает:
            Причину неудачи, пустую строку если файл записался.
        """
        import torch

        # Отсчёты вне диапазона от минус единицы до единицы переполняют int16.
        samples = (audio.clamp(-1.0, 1.0) * _MAX_AMPLITUDE).to(torch.int16)

        try:
            output_path.parent.mkdir(parents = True, exist_ok = True)
            with wave.open(str(output_path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(_SAMPLE_WIDTH_BYTES)
                target.setframerate(self._config.sample_rate)
                target.writeframes(samples.numpy().tobytes())
        except Exception as error:
            print(f"[speaking] звук не сохранился: {type(error).__name__}: {error}")
            return "файл со звуком не сохранился"

        return ""

    def _loaded_model(self) -> tuple[Any | None, str]:
        """
        Отдаёт модель, загружая её при первом обращении.

        Возвращает:
            Пару «модель, причина неудачи». При успехе причина пустая, при
            неудаче модель None.
        """
        if self._model is not None:
            return self._model, ""

        try:
            import torch
        except (ImportError, OSError) as error:
            # OSError наравне с ImportError: torch подгружает свои бинарники на
            # импорте и без них падает именно так.
            print(f"[speaking] библиотека torch недоступна: {type(error).__name__}: {error}")
            return None, "библиотека синтеза недоступна"

        if self._config.hub_directory is not None:
            self._config.hub_directory.mkdir(parents = True, exist_ok = True)
            torch.hub.set_dir(str(self._config.hub_directory))

        device = _resolve_device(torch = torch, device = self._config.device)
        print(f"[speaking] загрузка модели {self._config.model_id} ({device})")

        try:
            model, _example_text = torch.hub.load(
                repo_or_dir = _HUB_REPOSITORY,
                model = _HUB_MODEL,
                language = self._config.language,
                speaker = self._config.model_id,
                trust_repo = True,
            )
            model.to(torch.device(device))
        except Exception as error:
            print(f"[speaking] модель не загрузилась: {type(error).__name__}: {error}")
            return None, f"модель {self._config.model_id} не загрузилась"

        self._model = model
        return self._model, ""


def _render_ssml(text: str, settings: VoiceSettings) -> str:
    """
    Заворачивает текст в разметку ssml с темпом и высотой.

    Аргументы:
        text: что произнести.
        settings: голос, темп и высота.

    Возвращает:
        Строку ssml целиком, вместе с корневым тегом.
    """
    return (
        "<speak>"
        f'<prosody rate="{settings.rate}" pitch="{settings.pitch}">{escape(text)}</prosody>'
        "</speak>"
    )


def _resolve_device(torch: Any, device: str) -> str:
    """
    Выбирает устройство вычислений по настройке.

    Аргументы:
        torch: модуль torch.
        device: значение настройки: auto, cpu, cuda.

    Возвращает:
        Имя устройства, готовое для torch.device.
    """
    if device != "auto":
        return device

    return "cuda" if torch.cuda.is_available() else "cpu"
