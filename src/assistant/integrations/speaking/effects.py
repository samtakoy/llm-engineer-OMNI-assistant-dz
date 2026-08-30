"""
Реестр звуковых эффектов поверх torchaudio.functional.

effect_catalog отдаёт имена эффектов с описанием звучания, apply_effect
накладывает эффект на готовый звук. Сила эффекта задаётся меткой, числа за
метками лежат здесь.

Torch и torchaudio импортируются внутри функций.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Числа за метками силы эффекта.
_STRENGTH_SCALE = {
    "low": 0.35,
    "medium": 0.65,
    "high": 1.0,
}

# Множители глубины эффектов: единица - формула в полную силу, половина - вдвое
# слабее. Ручка подстройки эффекта на слух.
_GROWL_DEPTH = 0.33
_CARTOON_DEPTH = 0.5
_GHOST_DEPTH = 0.4
_GIANT_DEPTH = 0.2

# Частота дрожания голоса в рычании, герцы.
_GROWL_RATTLE_HZ = 32.0
# Насколько глубоко дрожание проваливает громкость при полной силе.
_GROWL_RATTLE_SHARE = 0.5
# Доля копии на октаву ниже в рычании при полной силе.
_GROWL_SUBHARMONIC_SHARE = 0.6
# На сколько ускоряется речь мультяшного голоса при полной силе.
_CARTOON_SPEED_RANGE = 0.35
# Длина окна преобразования Фурье для шёпота призрака, отсчёты.
_GHOST_WINDOW = 1024
# Усиление шёпота: после случайной фазы звук тише исходного.
_GHOST_WHISPER_GAIN = 1.6
# Сила эха зала, которое достаётся призраку.
_GHOST_ROOM_SHARE = 0.4
# Сдвиг голоса великана вниз при полной силе, полутоны.
_GIANT_PITCH_RANGE = 6.0
# Сила эха зала, которое достаётся великану.
_GIANT_ROOM_SHARE = 0.8

# Уровень, ниже которого звук считается тишиной и не выравнивается.
_SILENCE_LEVEL = 1e-6

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
    Собирает рычащий бас: голос ниже, под ним октава, поверх дрожание связок.

    Октава ниже и медленная амплитудная модуляция дают скрип живого горла,
    перегруз добавляет только грязь по краям.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    import torch
    from torchaudio.functional import overdrive, pitch_shift

    depth = strength * _GROWL_DEPTH
    lowered = pitch_shift(
        waveform = audio,
        sample_rate = sample_rate,
        n_steps = -(1.0 + 4.0 * depth),
    )
    subharmonic = pitch_shift(
        waveform = lowered,
        sample_rate = sample_rate,
        n_steps = -12.0,
    )
    subharmonic_share = _GROWL_SUBHARMONIC_SHARE * depth
    voiced = lowered * (1.0 - subharmonic_share) + subharmonic * subharmonic_share

    seconds = torch.arange(voiced.shape[-1]) / sample_rate
    swing = 0.5 - 0.5 * torch.cos(2.0 * math.pi * _GROWL_RATTLE_HZ * seconds)
    rattle = 1.0 - _GROWL_RATTLE_SHARE * depth * swing

    return _matched_loudness(
        processed = overdrive(
            waveform = voiced * rattle,
            gain = 2.0 + 8.0 * depth,
            colour = 20.0,
        ),
        source = audio,
    )


def _apply_cartoon(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Ускоряет запись: голос выше и суетливее, как в мультфильме.

    Пересчёт частоты дискретизации тянет вверх и высоту, и форманты, поэтому
    голос звучит цельным маленьким существом, а не сдавленным человеком. Речь
    при этом становится короче.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    from torchaudio.functional import speed

    raised, _ = speed(
        waveform = audio,
        orig_freq = sample_rate,
        factor = 1.0 + _CARTOON_SPEED_RANGE * strength * _CARTOON_DEPTH,
    )
    return _matched_loudness(processed = raised, source = audio)


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
    return _matched_loudness(
        processed = audio * (1.0 - wet_share) + wet * wet_share,
        source = audio,
    )


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
    return _matched_loudness(
        processed = overdrive(waveform = narrowed, gain = 5.0 * strength, colour = 20.0),
        source = audio,
    )


def _apply_ghost(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Стирает высоту голоса и оставляет шелест: шёпот призрака в пустом зале.

    Спектр раскладывается по окнам, громкость каждой частоты остаётся, фаза
    заменяется случайной. Тон пропадает, разборчивость остаётся - так устроен
    настоящий шёпот. Сверху идёт эхо зала.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    import torch

    window = torch.hann_window(_GHOST_WINDOW)
    hop_length = _GHOST_WINDOW // 4
    spectrum = torch.stft(
        input = audio,
        n_fft = _GHOST_WINDOW,
        hop_length = hop_length,
        window = window,
        return_complex = True,
    )
    random_phase = torch.rand_like(spectrum.abs()) * 2.0 * math.pi
    whispered = torch.istft(
        input = torch.polar(spectrum.abs(), random_phase),
        n_fft = _GHOST_WINDOW,
        hop_length = hop_length,
        window = window,
        length = audio.shape[-1],
    )

    whisper_share = min(1.0, strength * _GHOST_DEPTH)
    voice = audio * (1.0 - whisper_share) + whispered * _GHOST_WHISPER_GAIN * whisper_share
    return _apply_cave(
        audio = _matched_loudness(processed = voice, source = audio),
        sample_rate = sample_rate,
        strength = _GHOST_ROOM_SHARE * strength,
    )


def _apply_giant(audio: Any, sample_rate: int, strength: float) -> Any:
    """
    Опускает голос и ставит его в большой зал: голос громады.

    Отличие от рычания - нет перегруза и субгармоники, голос остаётся чистым,
    просто идёт снизу и издалека.

    Аргументы:
        audio: отсчёты звука.
        sample_rate: частота дискретизации.
        strength: сила эффекта от нуля до единицы.

    Возвращает:
        Обработанные отсчёты.
    """
    from torchaudio.functional import pitch_shift

    lowered = pitch_shift(
        waveform = audio,
        sample_rate = sample_rate,
        n_steps = -_GIANT_PITCH_RANGE * strength * _GIANT_DEPTH,
    )
    return _matched_loudness(
        processed = _apply_cave(
            audio = lowered,
            sample_rate = sample_rate,
            strength = _GIANT_ROOM_SHARE * strength,
        ),
        source = audio,
    )


def _matched_loudness(processed: Any, source: Any) -> Any:
    """
    Приводит громкость обработанного звука к громкости исходного.

    Среднеквадратичный уровень обработанных отсчётов подгоняется под уровень
    исходных, после чего пик ограничивается единицей. Звук тише порога тишины
    остаётся без выравнивания.

    Аргументы:
        processed: отсчёты после наложения эффекта.
        source: отсчёты до наложения эффекта.

    Возвращает:
        Отсчёты с громкостью исходного звука и пиком не больше единицы.
    """
    source_level = source.pow(2).mean().sqrt()
    processed_level = processed.pow(2).mean().sqrt()
    if source_level <= _SILENCE_LEVEL or processed_level <= _SILENCE_LEVEL:
        return _normalized(processed)

    return _normalized(processed * (source_level / processed_level))


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
        description = "рычащий бас: голос ниже, с октавой и дрожью",
        apply = _apply_growl,
    ),
    "cartoon": Effect(
        description = "мультяшный писк: голос выше и речь быстрее",
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
    "ghost": Effect(
        description = "шёпот призрака: голос без тона, эхом в пустоте",
        apply = _apply_ghost,
    ),
    "giant": Effect(
        description = "голос громады: чистый низ и большой зал",
        apply = _apply_giant,
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
