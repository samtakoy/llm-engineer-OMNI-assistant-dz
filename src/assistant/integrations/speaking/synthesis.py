"""
Синтез речи моделью silero и запись звука в wav.

SpeechSynthesizer отдаёт список голосов модели и озвучивает текст заданным
голосом, темпом, высотой и звуковым эффектом. Модель грузится при первой
озвучке и остаётся в поле объекта.

Кусок речи собирается из частей: каждая часть звучит своим голосом, между
частями кладётся тишина, всё вместе ложится в один файл.

Разметка в тексте и ненейтральные темп с высотой уходят в модель как ssml,
чистый текст с нейтральными настройками - как текст с ударениями и буквой ё:
в режиме ssml silero флаги ударений не принимает. Перед отправкой разметка
чистится санитайзером, абзацы и предложения расставляются по тексту.

При неудаче возвращается исход с причиной, исключения наружу не уходят.
"""

import logging
import wave
from pathlib import Path
from typing import Any

from .config import SpeakingConfig
from .effects import apply_effect
from .markup import drop_markup, sanitize_markup, wrap_speech_parts
from .outcomes import SynthesisOutcome
from .voices import VoiceSettings

logger = logging.getLogger(__name__)


# Откуда torch.hub берёт код silero и файл весов.
_HUB_REPOSITORY = "snakers4/silero-models"
_HUB_MODEL = "silero_tts"

# Ширина отсчёта: пишем в int16, так же его понимает wave и любой декодер.
_SAMPLE_WIDTH_BYTES = 2
_MAX_AMPLITUDE = 32767

# Темп и высота, при которых разметка не нужна.
_NEUTRAL_PROSODY = "medium"

# Тишина между соседними частями куска речи.
_PART_GAP_SECONDS = 0.35


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
        return self.synthesize_parts(parts = [(text, settings)], output_path = output_path)

    def synthesize_parts(
        self,
        parts: list[tuple[str, VoiceSettings]],
        output_path: Path,
    ) -> SynthesisOutcome:
        """
        Озвучивает части своими голосами, склеивает их и кладёт звук в один файл.

        Между соседними частями кладётся тишина. Часть, которая не озвучилась,
        пропускается с записью причины в журнал. Файл не пишется, только когда не
        озвучилась ни одна часть.

        Аргументы:
            parts: пары «текст, настройки голоса» в порядке произнесения.
            output_path: файл, куда писать звук.

        Возвращает:
            Исход озвучки: путь к файлу либо причина неудачи.
        """
        import torch

        model, load_error = self._loaded_model()
        if model is None:
            return SynthesisOutcome(path = None, error = load_error, seconds = 0.0)

        chunks: list[Any] = []
        last_error = "озвучивать нечего"

        for text, settings in parts:
            audio, render_error = self._rendered_part(
                model = model,
                text = text,
                settings = settings,
            )
            if audio is None:
                logger.warning(f"[speaking] часть пропущена: {render_error}")
                last_error = render_error
                continue

            if chunks:
                chunks.append(self._silence(like = audio))
            chunks.append(audio)

        if not chunks:
            return SynthesisOutcome(path = None, error = last_error, seconds = 0.0)

        audio = torch.cat(chunks)
        seconds = len(audio) / self._config.sample_rate

        write_error = self._write_wav(audio = audio, output_path = output_path)
        if write_error:
            return SynthesisOutcome(path = None, error = write_error, seconds = seconds)

        return SynthesisOutcome(path = output_path, error = "", seconds = seconds)

    def _rendered_part(
        self,
        model: Any,
        text: str,
        settings: VoiceSettings,
    ) -> tuple[Any | None, str]:
        """
        Озвучивает одну часть куска и накладывает на неё звуковой эффект.

        Аргументы:
            model: загруженная модель silero.
            text: что произнести.
            settings: голос, темп, высота и звуковой эффект.

        Возвращает:
            Пару «отсчёты звука, причина неудачи». При неудаче отсчёты None.
        """
        spoken_text = text.strip()
        if not spoken_text:
            return None, "озвучивать нечего"

        if settings.speaker not in model.speakers:
            return None, f"голоса {settings.speaker} нет в модели {self._config.model_id}"

        plain_text = drop_markup(text = spoken_text)
        audio, synthesis_error = self._synthesized_audio(
            model = model,
            plain_text = plain_text,
            marked_text = spoken_text,
            settings = settings,
        )
        if audio is None:
            return None, synthesis_error

        audio, effect_error = apply_effect(
            audio = audio,
            sample_rate = self._config.sample_rate,
            effect = settings.effect,
            strength = settings.effect_strength,
        )
        if effect_error:
            return None, effect_error

        return audio, ""

    def _silence(self, like: Any) -> Any:
        """
        Строит тишину, которая ложится между соседними частями куска.

        Аргументы:
            like: отсчёты соседней части; у тишины тот же тип отсчётов.

        Возвращает:
            Отсчёты тишины длиной _PART_GAP_SECONDS.
        """
        import torch

        return torch.zeros(int(self._config.sample_rate * _PART_GAP_SECONDS), dtype = like.dtype)

    def _synthesized_audio(
        self,
        model: Any,
        plain_text: str,
        marked_text: str,
        settings: VoiceSettings,
    ) -> tuple[Any | None, str]:
        """
        Озвучивает текст, спускаясь по ступеням разметки до первой удачной.

        Ступени: разметка режиссёра вместе с настройками голоса, потом одни
        настройки голоса без разметки, потом чистый текст. Каждая следующая
        беднее предыдущей, поэтому падение разбора ssml стоит пауз, но не темпа
        и высоты.

        Аргументы:
            model: загруженная модель silero.
            plain_text: что произнести без разметки.
            marked_text: тот же текст с разметкой режиссёра.
            settings: голос, темп, высота и звуковой эффект.

        Возвращает:
            Пару «отсчёты звука, причина неудачи». При неудаче отсчёты None.
        """
        last_error = ""

        for description, ssml in self._synthesis_steps(
            plain_text = plain_text,
            marked_text = marked_text,
            settings = settings,
        ):
            logger.info(f"[speaking] в синтез ({description}): {ssml if ssml is not None else plain_text}")
            try:
                return self._apply_tts(
                    model = model,
                    text = plain_text,
                    ssml = ssml,
                    settings = settings,
                ), ""
            except Exception as error:
                logger.warning(
                    f"[speaking] ступень «{description}» не удалась: "
                    f"{type(error).__name__}: {error}"
                )
                last_error = f"озвучка оборвалась: {type(error).__name__}"

        return None, last_error

    def _synthesis_steps(
        self,
        plain_text: str,
        marked_text: str,
        settings: VoiceSettings,
    ) -> list[tuple[str, str | None]]:
        """
        Собирает ступени озвучки от самой богатой к самой бедной.

        Аргументы:
            plain_text: что произнести без разметки.
            marked_text: тот же текст с разметкой режиссёра.
            settings: голос, темп и высота.

        Возвращает:
            Пары «название ступени, разметка ssml». У последней ступени разметки
            нет: там синтез идёт чистым текстом.
        """
        with_markup = self._prepare_ssml(text = marked_text, settings = settings)
        without_markup = self._prepare_ssml(text = plain_text, settings = settings)

        steps: list[tuple[str, str | None]] = []
        if with_markup is not None:
            steps.append(("разметка и настройки голоса", with_markup))
        if without_markup is not None and without_markup != with_markup:
            steps.append(("настройки голоса без разметки", without_markup))
        steps.append(("чистый текст", None))

        return steps

    def _prepare_ssml(self, text: str, settings: VoiceSettings) -> str | None:
        """
        Готовит разметку ssml, если она нужна.

        Аргументы:
            text: что произнести, с разметкой или без неё.
            settings: голос, темп и высота.

        Возвращает:
            Строку ssml либо None, когда хватает чистого текста: разметки в
            тексте нет, темп и высота нейтральные.
        """
        body, has_markup = sanitize_markup(text = text)
        is_neutral = settings.rate == _NEUTRAL_PROSODY and settings.pitch == _NEUTRAL_PROSODY

        if is_neutral and not has_markup:
            return None

        return _render_ssml(body = wrap_speech_parts(body = body), settings = settings)

    def _apply_tts(self, model: Any, text: str, ssml: str | None, settings: VoiceSettings) -> Any:
        """
        Зовёт синтез: чистым текстом либо готовой разметкой ssml.

        Аргументы:
            model: загруженная модель silero.
            text: что произнести без разметки.
            ssml: готовая разметка; None - озвучивать чистым текстом.
            settings: голос, темп и высота.

        Возвращает:
            Отсчёты звука тензором в диапазоне от минус единицы до единицы.
        """
        if ssml is None:
            return model.apply_tts(
                text = text,
                speaker = settings.speaker,
                sample_rate = self._config.sample_rate,
                put_accent = self._config.put_accent,
                put_yo = self._config.put_yo,
            )

        return model.apply_tts(
            ssml_text = ssml,
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
            logger.warning(f"[speaking] звук не сохранился: {type(error).__name__}: {error}")
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
            logger.warning(f"[speaking] библиотека torch недоступна: {type(error).__name__}: {error}")
            return None, "библиотека синтеза недоступна"

        if self._config.hub_directory is not None:
            self._config.hub_directory.mkdir(parents = True, exist_ok = True)
            torch.hub.set_dir(str(self._config.hub_directory))

        device = _resolve_device(torch = torch, device = self._config.device)
        logger.info(f"[speaking] загрузка модели {self._config.model_id} ({device})")

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
            logger.warning(f"[speaking] модель не загрузилась: {type(error).__name__}: {error}")
            return None, f"модель {self._config.model_id} не загрузилась"

        _fix_external_alphabet(model = model)

        self._model = model
        return self._model, ""


def _fix_external_alphabet(model: Any) -> None:
    """
    Ставит пустой словарь замен там, где модель хранит None.

    На пути ssml модель зовёт convert_to_orig без проверки ext_alph на None, и
    латинская буква в тексте роняет разбор. Пустой словарь включает ту же ветку
    «ничего не заменять», что стоит на пути чистого текста. Словарь лежит не в
    самой модели, а в её пакетах голосов, поэтому чинить надо каждый.

    Аргументы:
        model: загруженная модель silero.

    Возвращает:
        Ничего.
    """
    packages = getattr(model, "packages", [])
    for package in [*packages, model]:
        if getattr(package, "ext_alph", None) is None:
            package.ext_alph = {}


def _render_ssml(body: str, settings: VoiceSettings) -> str:
    """
    Заворачивает готовое тело в разметку ssml с темпом и высотой.

    Аргументы:
        body: тело ssml после санитайзера.
        settings: голос, темп и высота.

    Возвращает:
        Строку ssml целиком, вместе с корневым тегом.
    """
    return (
        "<speak>"
        f'<prosody rate="{settings.rate}" pitch="{settings.pitch}">{body}</prosody>'
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
