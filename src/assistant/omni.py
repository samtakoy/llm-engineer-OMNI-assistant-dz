"""
Сборка всех этапов: фотография персонажа, вопрос, экскурсия его голосом.

Точки входа зовут только эту функцию: командная строка и интерфейс идут одним
путём. Длительности этапов копит Stopwatch и возвращает рядом с ответом.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI

from assistant.graph import Answer, ResearchNotes, describe_nodes, new_run_id, run_research
from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.integrations.speaking import (
    NO_EFFECT,
    SpeechSynthesizer,
    VoiceSettings,
)
from assistant.logs import (
    log_look,
    log_markup,
    log_markup_skipped,
    log_narrator_style,
    log_persona,
    log_speech_failed,
    log_spoken,
    log_timing,
    log_voice,
    log_voice_fallback,
    log_voices_missing,
)
from assistant.observability import trace_run
from assistant.persona import (
    Persona,
    PersonaMode,
    build_narrator_style,
    build_persona,
    describe_look,
    mark_up_speech,
    pick_voice,
    render_narrator_prompt,
)
from assistant.timing import Stopwatch
from assistant.variables import (
    ENABLE_ALL_REASONING,
    SPEAKING_CONFIG,
    SPOKEN_PATH,
    VISION_MODEL,
    VISION_PROVIDER,
)


@dataclass(frozen = True)
class OmniOutcome:
    """
    Исход прогона.

    Атрибуты:
        answer: итоговый текст экскурсии; None при неудаче.
        notes: фактическая опора, на которой построен текст; None при неудаче.
        persona: рассказчик полями; заполняется только в режиме STRUCTURED.
        narrator_prompt: блок про рассказчика, ушедший в узел изложения; пустая
            строка, если рассказчик не задан.
        look: описание облика с фотографии; пустая строка, если фотографии не было.
        trace_path: файл журнала прогона; None, если журнал выключен.
        voice: настройки голоса; None, если озвучки не было.
        audio_paths: файлы с озвучкой по кускам, в порядке произнесения.
        timing: замеры длительности этапов.
        error: причина неудачи; пустая строка при успехе.
    """

    answer: Answer | None
    notes: ResearchNotes | None
    persona: Persona | None
    narrator_prompt: str
    look: str
    trace_path: Path | None
    voice: VoiceSettings | None
    audio_paths: list[Path]
    timing: Stopwatch
    error: str


def build_narrator_from_image(
    image_path: Path,
    persona_mode: PersonaMode,
    timing: Stopwatch,
) -> tuple[str, Persona | None, str, str]:
    """
    Строит блок про рассказчика по фотографии персонажа.

    Аргументы:
        image_path: файл с фотографией персонажа.
        persona_mode: способ сборки: одной фразой либо полями схемы.
        timing: копилка замеров.

    Возвращает:
        Четвёрку «блок про рассказчика, персонаж полями, описание облика,
        причина неудачи». Персонаж заполняется только в режиме STRUCTURED.
        При неудаче блок пустой.
    """
    vision_llm = build_llm(
        role = NodeRole.VISION,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = VISION_MODEL,
        provider = VISION_PROVIDER,
    )

    with timing.stage(name = "облик"):
        look, error = describe_look(llm = vision_llm, image_path = image_path)
    if error:
        return "", None, "", error

    log_look(look = look)

    writing_llm = build_llm(
        role = NodeRole.WRITING,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = None,
    )

    if persona_mode is PersonaMode.FREE:
        # свободное изложение описания
        with timing.stage(name = "рассказчик"):
            style, error = build_narrator_style(llm = writing_llm, look = look)
        if error:
            return "", None, look, error

        log_narrator_style(style = style)
        return style, None, look, ""

    # описание рассказчика в виде структурированного разложения по осям
    with timing.stage(name = "рассказчик"):
        persona, error = build_persona(llm = writing_llm, look = look)
    if error:
        return "", None, look, error

    log_persona(persona = persona)
    return render_narrator_prompt(persona = persona), persona, look, ""


def default_voice(speakers: list[str]) -> VoiceSettings:
    """
    Собирает настройки голоса для прогона без персонажа.

    Аргументы:
        speakers: имена голосов, которые знает модель синтеза.

    Возвращает:
        Первый голос списка, нейтральные темп и высоту, без эффекта.
    """
    return VoiceSettings(
        speaker = speakers[0],
        rate = "medium",
        pitch = "medium",
        effect = NO_EFFECT,
        effect_strength = "medium",
    )


def split_into_pieces(answer: Answer) -> list[str]:
    """
    Режет ответ на куски для озвучки.

    Аргументы:
        answer: итоговый текст.

    Возвращает:
        Вступление, разделы вместе с заголовками и завершение, в порядке чтения.
    """
    pieces = [answer.intro]
    pieces.extend(f"{section.title}. {section.content}" for section in answer.sections)
    pieces.append(answer.closing)

    return [piece for piece in pieces if piece.strip()]


def speak_answer(
    answer: Answer,
    persona: Persona | None,
    settings: VoiceSettings,
    synthesizer: SpeechSynthesizer,
    llm: ChatOpenAI,
    is_markup_on: bool,
    timing: Stopwatch,
) -> list[Path]:
    """
    Озвучивает ответ по кускам, печатая каждый готовый файл сразу.

    Аргументы:
        answer: итоговый текст.
        persona: рассказчик полями; None - разметка не запрашивается.
        settings: настройки голоса.
        synthesizer: синтезатор речи.
        llm: клиент текстовой модели для разметки.
        is_markup_on: просить модель разметить текст перед озвучкой.
        timing: копилка замеров.

    Возвращает:
        Файлы с озвучкой в порядке произнесения. Неудачные куски пропускаются.
    """
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    paths: list[Path] = []

    for index, piece in enumerate(split_into_pieces(answer = answer), start = 1):
        spoken_text = piece

        if is_markup_on and persona is not None:
            with timing.stage(name = f"разметка {index}"):
                marked, error = mark_up_speech(llm = llm, persona = persona, text = piece)
            if error:
                log_markup_skipped(index = index, reason = error)
            else:
                spoken_text = marked
                log_markup(index = index, marked_text = marked)

        with timing.stage(name = f"озвучка {index}"):
            outcome = synthesizer.synthesize(
                text = spoken_text,
                settings = settings,
                output_path = SPOKEN_PATH / f"{stamp}-{index:02d}.wav",
            )

        if outcome.error:
            log_speech_failed(index = index, reason = outcome.error)
            continue

        log_spoken(index = index, path = outcome.path, seconds = outcome.seconds)
        paths.append(outcome.path)

    return paths


def run_omni_assistant(
    image_path: Path | None,
    narrator_style: str | None,
    persona_mode: PersonaMode,
    question: str,
    is_speech_on: bool,
    is_markup_on: bool,
) -> OmniOutcome:
    """
    Проводит экскурсию по вопросу от лица заданного рассказчика.

    Рассказчик берётся из фотографии либо из готовой фразы. Без того и другого
    текст пишется обычным рассказчиком. Озвучка идёт кусками: вступление,
    разделы, завершение.

    Аргументы:
        image_path: файл с фотографией персонажа; None - без фотографии.
        narrator_style: готовая фраза про голос рассказчика; None - без неё.
        persona_mode: способ сборки рассказчика по фотографии.
        question: вопрос пользователя.
        is_speech_on: озвучивать готовый текст.
        is_markup_on: размечать текст перед озвучкой.

    Возвращает:
        Исход прогона: текст экскурсии с опорой и озвучкой либо причина неудачи.
    """
    timing = Stopwatch()
    run_id = new_run_id()

    with trace_run(trace_id = run_id, node_rows = describe_nodes(), origin_rows = []) as trace:
        trace_path = trace.path() if trace is not None else None
        persona: Persona | None = None
        narrator_prompt = ""
        look = ""

        if narrator_style is not None:
            narrator_prompt = narrator_style
            log_narrator_style(style = narrator_prompt)
        elif image_path is not None:
            narrator_prompt, persona, look, error = build_narrator_from_image(
                image_path = image_path,
                persona_mode = persona_mode,
                timing = timing,
            )
            if error:
                log_timing(timing = timing)
                return OmniOutcome(
                    answer = None,
                    notes = None,
                    persona = None,
                    narrator_prompt = "",
                    look = look,
                    trace_path = trace_path,
                    voice = None,
                    audio_paths = [],
                    timing = timing,
                    error = error,
                )

        with timing.stage(name = "ресёрч"):
            answer, notes = run_research(
                question = question,
                narrator_prompt = narrator_prompt or None,
                run_id = run_id,
                callbacks = [trace] if trace is not None else [],
            )

        voice: VoiceSettings | None = None
        audio_paths: list[Path] = []

        if is_speech_on:
            voice, audio_paths = speak_outcome(
                answer = answer,
                persona = persona,
                is_markup_on = is_markup_on,
                timing = timing,
            )

        log_timing(timing = timing)

        return OmniOutcome(
            answer = answer,
            notes = notes,
            persona = persona,
            narrator_prompt = narrator_prompt,
            look = look,
            trace_path = trace_path,
            voice = voice,
            audio_paths = audio_paths,
            timing = timing,
            error = "",
        )


def speak_outcome(
    answer: Answer,
    persona: Persona | None,
    is_markup_on: bool,
    timing: Stopwatch,
) -> tuple[VoiceSettings | None, list[Path]]:
    """
    Подбирает голос и озвучивает готовый текст.

    Аргументы:
        answer: итоговый текст.
        persona: рассказчик полями; None - голос берётся по умолчанию.
        is_markup_on: размечать текст перед озвучкой.
        timing: копилка замеров.

    Возвращает:
        Пару «настройки голоса, файлы с озвучкой». При неудаче настройки None,
        список пустой.
    """
    synthesizer = SpeechSynthesizer(config = SPEAKING_CONFIG)

    with timing.stage(name = "загрузка синтеза"):
        speakers, error = synthesizer.available_speakers()
    if error:
        log_voices_missing(reason = error)
        return None, []

    writing_llm = build_llm(
        role = NodeRole.WRITING,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = None,
    )

    if persona is None:
        settings = default_voice(speakers = speakers)
    else:
        extraction_llm = build_llm(
            role = NodeRole.EXTRACTION,
            is_reasoning_forced = ENABLE_ALL_REASONING,
            model = None,
        )
        with timing.stage(name = "подбор голоса"):
            settings, error = pick_voice(
                llm = extraction_llm,
                persona = persona,
                speakers = speakers,
            )
        if error:
            log_voice_fallback(reason = error)
            settings = default_voice(speakers = speakers)

    log_voice(settings = settings)

    paths = speak_answer(
        answer = answer,
        persona = persona,
        settings = settings,
        synthesizer = synthesizer,
        llm = writing_llm,
        is_markup_on = is_markup_on,
        timing = timing,
    )

    return settings, paths
