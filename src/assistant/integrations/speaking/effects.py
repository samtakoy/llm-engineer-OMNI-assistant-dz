"""
Реестр звуковых эффектов поверх torchaudio.functional.

effect_catalog отдаёт имена эффектов с описанием звучания, apply_effect
накладывает эффект на готовый звук. Сила эффекта задаётся меткой, числа за
метками лежат здесь.

Torch и torchaudio импортируются внутри функций.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Числа за метками силы эффекта.
_STRENGTH_SCALE = {
    "low": 0.35,
    "medium": 0.65,
    "high": 1.0,
}

# Имя эффекта, который ничего не меняет.
NO_EFFECT = "none"


@dataclass(frozen = True)
class Effect:
    """
    Эффект реестра.

    Атрибуты:
        description: как эффект меняет звучание; уходит в промпт подбора голоса.
        apply: наложение эффекта на отсчёты звука.
    """

    description: str
    apply: Callable[[Any, int, float], Any]


def _apply_none(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Отдаёт звук без изменений.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта; не используется.

    Возвращает:
        Те же отсчёты.
    """
    return audio


def _apply_growl(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Опускает голос вниз и добавляет перегруз: рычащий бас.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    from torchaudio.functional import overdrive, pitch_shift

    lowered = pitch_shift(
        waveform = audio,
        sample_rate = sample_rate,
        n_steps = -(1.0 + 4.0 * strength),
    )
    return overdrive(waveform = lowered, gain = 5.0 + 25.0 * strength, colour = 20.0)


def _apply_cartoon(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Поднимает голос вверх: мультяшное звучание.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    from torchaudio.functional import pitch_shift

    return pitch_shift(
        waveform = audio,
        sample_rate = sample_rate,
        n_steps = 1.0 + 4.0 * strength,
    )


def _apply_cave(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Подмешивает эхо каменного зала.

    Отклик - затухающий шум, длина отклика растёт с силой эффекта.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    import torch
    from torchaudio.functional import fftconvolve

    tail_length = int(sample_rate * (0.15 + 0.45 * strength))
    decay = torch.exp(-torch.linspace(0.0, 6.0, tail_length))
    response = torch.randn(tail_length) * decay
    # Первый отсчёт отклика - сам голос, остальное хвост зала.
    response[0] = 1.0

    wet = fftconvolve(audio, response, mode = "same")
    wet_share = 0.5 * strength
    return _normalized(audio * (1.0 - wet_share) + wet * wet_share)


def _apply_radio(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Срезает низ и верх: голос из динамика.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    from torchaudio.functional import highpass_biquad, lowpass_biquad, overdrive

    narrowed = highpass_biquad(
        waveform = audio,
        sample_rate = sample_rate,
        cutoff_freq = 300.0 + 300.0 * strength,
    )
    narrowed = lowpass_biquad(
        waveform = narrowed,
        sample_rate = sample_rate,
        cutoff_freq = 3400.0 - 1000.0 * strength,
    )
    return _normalized(overdrive(waveform = narrowed, gain = 5.0 * strength, colour = 20.0))


def _normalized(audio: Any) -> Any:
    """
    Приводит пик отсчётов к единице, если он её перерос.

    Аргументы:
        audio: отсчёты звука.

    Возвращает:
        Отсчёты, у которых пик не больше единицы.
    """
    peak = audio.abs().max()
    if peak <= 1.0:
        return audio

    return audio / peak


_EFFECTS = {
    NO_EFFECT: Effect(
        description = "чистый голос без обработки",
        apply = _apply_none,
    ),
    "growl": Effect(
        description = "рычащий бас: голос ниже и с перегрузом",
        apply = _apply_growl,
    ),
    "cartoon": Effect(
        description = "мультяшный писк: голос выше",
        apply = _apply_cartoon,
    ),
    "cave": Effect(
        description = "эхо каменного зала или пещеры",
        apply = _apply_cave,
    ),
    "radio": Effect(
        description = "голос из динамика: узкая полоса и хрип",
        apply = _apply_radio,
    ),
}


def available_effects() -> list[str]:
    """
    Отдаёт имена эффектов реестра.

    Возвращает:
        Имена эффектов в порядке объявления.
    """
    return list(_EFFECTS)


def effect_catalog() -> dict[str, str]:
    """
    Отдаёт описания эффектов по именам.

    Возвращает:
        Описание звучания для каждого имени эффекта.
    """
    return {name: effect.description for name, effect in _EFFECTS.items()}


def apply_effect(audio: Any, sample_rate: int, effect: str, strength: str) -> tuple[Any, str]:
    """
    Накладывает эффект реестра на отсчёты звука.

    Аргументы:
        audio: отсчёты звука от минус единицы до единицы.
        sample_rate: частота дискретизации.
        effect: имя эффекта из реестра.
        strength: сила эффекта меткой: low, medium, high.

    Возвращает:
        Пару «обработанные отсчёты, причина неудачи». При неудаче возвращаются
        исходные отсчёты.
    """
    if effect not in _EFFECTS:
        return audio, f"эффекта {effect} нет в реестре"

    if strength not in _STRENGTH_SCALE:
        return audio, f"силы эффекта {strength} нет в шкале"

    try:
        processed = _EFFECTS[effect].apply(audio, sample_rate, _STRENGTH_SCALE[strength])
    except Exception as error:
        logger.warning(f"[speaking] эффект {effect} не наложился: {type(error).__name__}: {error}")
        return audio, f"эффект {effect} не наложился: {type(error).__name__}"

    return processed, ""
