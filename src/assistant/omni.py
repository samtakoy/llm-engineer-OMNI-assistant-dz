"""
Сборка всех этапов: вопрос, фотография персонажа, экскурсия его голосом.

Вопрос берётся текстом, из файла с записью или с микрофона. Записанный прогон
продолжается с указанного узла теми же этапами. Факты записанного прогона можно
взять готовыми и пересобрать по ним только рассказчика и текст. Длительности
этапов копит Stopwatch и возвращает рядом с ответом.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from assistant.documents import write_run_documents
from assistant.graph import Answer, ResearchNotes
from assistant.graph_runs import (
    ResearchStep,
    build_node_llms,
    describe_nodes,
    find_resume_point,
    new_run_id,
    resume_research,
    resume_research_staged,
    run_research_staged,
)
from assistant.integrations.listening import SILENCE_LEVEL, SpeechRecognizer, record
from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.integrations.speaking import (
    NO_EFFECT,
    SpeechSynthesizer,
    VoiceSettings,
)
from assistant.logs import (
    log_document,
    log_look,
    log_markup,
    log_markup_skipped,
    log_narrator_style,
    log_persona,
    log_question,
    log_recording,
    log_recording_silent,
    log_speech_failed,
    log_spoken,
    log_timing,
    log_title_voice,
    log_title_voice_missing,
    log_voice,
    log_voice_fallback,
    log_voices_missing,
)
from assistant.observability import trace_run
from assistant.persona import (
    UNKNOWN_GENDER,
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
    FEMALE_SPEAKER,
    LISTENING_CONFIG,
    MALE_SPEAKER,
    SPEAKING_CONFIG,
    SPOKEN_PATH,
    VISION_MODEL,
    VISION_PROVIDER,
)

# Значение record_seconds, при котором запись идёт до нажатия Enter.
RECORD_UNTIL_ENTER = 0.0


@dataclass(frozen = True)
class OmniOutcome:
    """
    Исход прогона.

    Атрибуты:
        question: вопрос пользователя, распознанный или взятый текстом; пустая
            строка, если добыть его не вышло.
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

    question: str
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


@dataclass(frozen = True)
class OmniStage:
    """
    Снимок прогона после очередного этапа.

    Атрибуты:
        name: имя завершённого этапа.
        step: шаг графа, если этап - шаг ресёрча; иначе None.
        outcome: исход в том виде, в каком он собран на этот момент.
    """

    name: str
    step: ResearchStep | None
    outcome: OmniOutcome


def resolve_question(
    question_text: str | None,
    audio_path: Path | None,
    record_seconds: float | None,
    timing: Stopwatch,
) -> tuple[str, str]:
    """
    Достаёт вопрос из заданного источника: текстом, из файла или с микрофона.

    Аргументы:
        question_text: вопрос текстом; None - вопрос задан иначе.
        audio_path: файл с записью вопроса; None - вопрос задан иначе.
        record_seconds: сколько секунд писать с микрофона; RECORD_UNTIL_ENTER и
            меньше - до нажатия Enter; None - вопрос задан иначе.
        timing: копилка замеров.

    Возвращает:
        Пару «текст вопроса, причина неудачи». При неудаче текст пустой.
    """
    if question_text is not None:
        return question_text, ""

    if audio_path is not None:
        return transcribe_file(audio_path = audio_path, timing = timing)

    if record_seconds is None:
        return "", "источник вопроса не задан"

    with timing.stage(name = "запись"):
        outcome = record(
            seconds = None if record_seconds <= RECORD_UNTIL_ENTER else record_seconds,
            config = LISTENING_CONFIG,
        )

    if outcome.error or outcome.path is None:
        return "", f"записать не вышло: {outcome.error}"

    log_recording(path = outcome.path, seconds = outcome.seconds, peak_level = outcome.peak_level)
    if outcome.peak_level < SILENCE_LEVEL:
        log_recording_silent()

    return transcribe_file(audio_path = outcome.path, timing = timing)


def transcribe_file(audio_path: Path, timing: Stopwatch) -> tuple[str, str]:
    """
    Распознаёт вопрос из файла с записью.

    Аргументы:
        audio_path: файл с записью вопроса.
        timing: копилка замеров.

    Возвращает:
        Пару «текст вопроса, причина неудачи». При неудаче текст пустой.
    """
    recognizer = SpeechRecognizer(config = LISTENING_CONFIG)

    started = time.monotonic()
    outcome = recognizer.transcribe(audio_path = audio_path)
    spent_seconds = time.monotonic() - started

    if outcome.load_seconds > 0:
        timing.add(name = "загрузка модели речи", seconds = outcome.load_seconds)

    timing.add(name = "распознавание", seconds = spent_seconds - outcome.load_seconds)

    if outcome.error:
        return "", f"распознать не вышло: {outcome.error}"

    log_question(question = outcome.text, is_from_cache = outcome.from_cache)
    return outcome.text, ""


def build_narrator_from_image(
    image_path: Path,
    persona_mode: PersonaMode,
    timing: Stopwatch,
    callbacks: list[BaseCallbackHandler],
) -> tuple[str, Persona | None, str, str]:
    """
    Строит блок про рассказчика по фотографии персонажа.

    Аргументы:
        image_path: файл с фотографией персонажа.
        persona_mode: способ сборки: одной фразой либо полями схемы.
        timing: копилка замеров.
        callbacks: слушатели прогона; журнал заводит вызывающий.

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
        look, error = describe_look(
            llm = vision_llm,
            image_path = image_path,
            callbacks = callbacks,
        )
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
            style, error = build_narrator_style(
                llm = writing_llm,
                look = look,
                callbacks = callbacks,
            )
        if error:
            return "", None, look, error

        log_narrator_style(style = style)
        return style, None, look, ""

    # описание рассказчика в виде структурированного разложения по осям
    with timing.stage(name = "рассказчик"):
        persona, error = build_persona(
            llm = writing_llm,
            look = look,
            callbacks = callbacks,
        )
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


def speakers_by_gender() -> dict[str, str]:
    """
    Собирает голоса, чей пол проекту известен, по полу рассказчика.

    Возвращает:
        Пол рассказчика как ключ, имя голоса как значение.
    """
    return {
        "мужской": MALE_SPEAKER,
        "женский": FEMALE_SPEAKER,
    }


def title_voice(narrator_gender: str, speakers: list[str]) -> VoiceSettings | None:
    """
    Собирает настройки голоса диктора, читающего заголовки разделов.

    Пол диктора противоположен полу рассказчика: женскому рассказчику отвечает
    мужской голос, остальным - женский. Темп и высота у диктора средние, эффекта
    нет.

    Аргументы:
        narrator_gender: пол рассказчика.
        speakers: имена голосов, которые знает модель синтеза.

    Возвращает:
        Настройки голоса диктора либо None, когда такого голоса в модели нет.
    """
    speaker = MALE_SPEAKER if narrator_gender == "женский" else FEMALE_SPEAKER

    if speaker not in speakers:
        log_title_voice_missing(speaker = speaker)
        return None

    log_title_voice(speaker = speaker)
    return VoiceSettings(
        speaker = speaker,
        rate = "medium",
        pitch = "medium",
        effect = NO_EFFECT,
        effect_strength = "medium",
    )


@dataclass(frozen = True)
class SpokenPiece:
    """
    Кусок ответа для озвучки.

    Атрибуты:
        title: заголовок раздела; пустая строка у вступления и завершения.
        text: текст куска без заголовка.
    """

    title: str
    text: str


def split_into_pieces(answer: Answer) -> list[SpokenPiece]:
    """
    Режет ответ на куски для озвучки.

    Аргументы:
        answer: итоговый текст.

    Возвращает:
        Вступление, разделы вместе с заголовками и завершение, в порядке чтения.
        Заголовок лежит отдельно от текста раздела.
    """
    pieces = [SpokenPiece(title = "", text = answer.intro)]
    pieces.extend(
        SpokenPiece(title = section.title, text = section.content)
        for section in answer.sections
    )
    pieces.append(SpokenPiece(title = "", text = answer.closing))

    return [piece for piece in pieces if piece.text.strip()]


def speech_parts(
    title: str,
    spoken_text: str,
    settings: VoiceSettings,
    title_settings: VoiceSettings | None,
) -> list[tuple[str, VoiceSettings]]:
    """
    Собирает части одного куска речи с голосом на каждую.

    Аргументы:
        title: заголовок раздела; пустая строка - заголовка нет.
        spoken_text: текст куска, размеченный либо исходный.
        settings: настройки голоса персонажа.
        title_settings: настройки голоса диктора; None - заголовок читает голос
            персонажа.

    Возвращает:
        Пары «текст, настройки голоса» в порядке произнесения. У куска без
        заголовка часть одна.
    """
    if not title.strip():
        return [(spoken_text, settings)]

    return [
        (title, title_settings if title_settings is not None else settings),
        (spoken_text, settings),
    ]


def speak_answer_staged(
    answer: Answer,
    narrator_prompt: str,
    settings: VoiceSettings,
    title_settings: VoiceSettings | None,
    synthesizer: SpeechSynthesizer,
    llm: ChatOpenAI,
    is_markup_on: bool,
    stamp: str,
    timing: Stopwatch,
    callbacks: list[BaseCallbackHandler],
) -> Iterator[tuple[str, Path | None]]:
    """
    Озвучивает ответ по кускам, отдавая каждый готовый файл сразу.

    Заголовок раздела читает диктор, тело раздела - персонаж, обе озвучки
    ложатся в один файл. Разметку получает только тело.

    Аргументы:
        answer: итоговый текст.
        narrator_prompt: блок про рассказчика; пустая строка - разметка не
            запрашивается.
        settings: настройки голоса персонажа.
        title_settings: настройки голоса диктора; None - заголовок читает голос
            персонажа.
        synthesizer: синтезатор речи.
        llm: клиент текстовой модели для разметки.
        is_markup_on: просить модель разметить текст перед озвучкой.
        stamp: штамп времени прогона, общий с markdown прогона.
        timing: копилка замеров.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Пары «событие, файл с озвучкой» по одной, в порядке произнесения. Разметка
        объявляется до вызова модели и файла не несёт. Неудачные куски пропускаются.
    """
    for index, piece in enumerate(split_into_pieces(answer = answer), start = 1):
        spoken_text = piece.text

        if is_markup_on and narrator_prompt:
            yield f"разметка {index}", None

            with timing.stage(name = f"разметка {index}"):
                marked, error = mark_up_speech(
                    llm = llm,
                    narrator_prompt = narrator_prompt,
                    text = piece.text,
                    callbacks = callbacks,
                )
            if error:
                log_markup_skipped(index = index, reason = error)
            else:
                spoken_text = marked
                log_markup(index = index, marked_text = marked)

        with timing.stage(name = f"озвучка {index}"):
            outcome = synthesizer.synthesize_parts(
                parts = speech_parts(
                    title = piece.title,
                    spoken_text = spoken_text,
                    settings = settings,
                    title_settings = title_settings,
                ),
                output_path = SPOKEN_PATH / f"{stamp}-{index:02d}.wav",
            )

        if outcome.error:
            log_speech_failed(index = index, reason = outcome.error)
            continue

        log_spoken(index = index, path = outcome.path, seconds = outcome.seconds)
        yield f"озвучка {index}", outcome.path


def empty_outcome(trace_path: Path | None, timing: Stopwatch) -> OmniOutcome:
    """
    Заводит пустой исход прогона, который дальше наполняется по ходу этапов.

    Аргументы:
        trace_path: файл журнала прогона; None, если журнал выключен.
        timing: копилка замеров.

    Возвращает:
        Исход без вопроса, текста, рассказчика и озвучки.
    """
    return OmniOutcome(
        question = "",
        answer = None,
        notes = None,
        persona = None,
        narrator_prompt = "",
        look = "",
        trace_path = trace_path,
        voice = None,
        audio_paths = [],
        timing = timing,
        error = "",
    )


def failed_stage(outcome: OmniOutcome, error: str) -> OmniStage:
    """
    Заканчивает прогон неудачей: пишет замеры и собирает последний снимок.

    Аргументы:
        outcome: исход прогона на момент неудачи.
        error: причина неудачи.

    Возвращает:
        Снимок прогона с причиной неудачи.
    """
    log_timing(timing = outcome.timing)

    return OmniStage(name = "неудача", step = None, outcome = replace(outcome, error = error))


def finish_run_staged(
    outcome: OmniOutcome,
    is_speech_on: bool,
    is_markup_on: bool,
    callbacks: list[BaseCallbackHandler],
) -> Iterator[OmniStage]:
    """
    Пишет markdown прогона, озвучивает готовый текст и отдаёт снимок прогона
    после каждого события озвучки.

    Markdown ложится рядом с озвучкой одним с ней штампом времени и пишется
    независимо от озвучки.

    Аргументы:
        outcome: исход прогона с готовым текстом и фактической опорой.
        is_speech_on: озвучивать готовый текст.
        is_markup_on: размечать текст перед озвучкой.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Снимки прогона по одному: по снимку на каждое событие озвучки и последний
        снимок с полным исходом.
    """
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    for document_path in write_run_documents(
        question = outcome.question,
        answer = outcome.answer,
        look = outcome.look,
        persona = outcome.persona,
        narrator_prompt = outcome.narrator_prompt,
        directory = SPOKEN_PATH,
        stamp = stamp,
    ):
        log_document(path = document_path)

    if is_speech_on:
        for stage in speak_outcome_staged(
            outcome = outcome,
            is_markup_on = is_markup_on,
            stamp = stamp,
            callbacks = callbacks,
        ):
            outcome = stage.outcome
            yield stage

    log_timing(timing = outcome.timing)

    yield OmniStage(name = "готово", step = None, outcome = outcome)


def run_omni_assistant_staged(
    image_path: Path | None,
    narrator_style: str | None,
    persona_mode: PersonaMode,
    question_text: str | None,
    audio_path: Path | None,
    record_seconds: float | None,
    reuse_run_id: str | None,
    is_speech_on: bool,
    is_markup_on: bool,
) -> Iterator[OmniStage]:
    """
    Проводит экскурсию по вопросу, отдавая снимок прогона после каждого этапа.

    Вопрос берётся из одного из трёх источников: текстом, из файла с записью или
    с микрофона. Рассказчик берётся из фотографии либо из готовой фразы. Без
    того и другого текст пишется обычным рассказчиком. Озвучка идёт кусками:
    вступление, разделы, завершение.

    С переиспользованием фактов сбор пропускается: вопрос и опора берутся из
    записанного прогона, а рассказчик собирается заново по фотографии или фразе.

    Аргументы:
        image_path: файл с фотографией персонажа; None - без фотографии.
        narrator_style: готовая фраза про голос рассказчика; None - без неё.
        persona_mode: способ сборки рассказчика по фотографии.
        question_text: вопрос текстом; None - вопрос задан иначе.
        audio_path: файл с записью вопроса; None - вопрос задан иначе.
        record_seconds: сколько секунд писать с микрофона; RECORD_UNTIL_ENTER и
            меньше - до нажатия Enter; None - вопрос задан иначе.
        reuse_run_id: прогон, факты которого берутся готовыми; None - собирать
            факты заново.
        is_speech_on: озвучивать готовый текст.
        is_markup_on: размечать текст перед озвучкой.

    Возвращает:
        Снимки прогона по одному, от разобранного вопроса до готовой озвучки.
        Оборвавшийся прогон отдаёт последним снимок с причиной неудачи.
    """
    timing = Stopwatch()
    run_id = new_run_id()

    # У прогона на готовых фактах журнал свой: имя исходного прогона плюс время
    # рестарта. Исходный файл остаётся нетронутым, а происхождение видно и в
    # имени, и в шапке.
    trace_id = run_id if reuse_run_id is None else f"{reuse_run_id}+{run_id}"
    origin_rows = (
        []
        if reuse_run_id is None
        else [f"- факты прогона `{reuse_run_id}`, сбор пропущен"]
    )

    node_rows = describe_nodes(llms = build_node_llms())

    with trace_run(trace_id = trace_id, node_rows = node_rows, origin_rows = origin_rows) as trace:
        trace_path = trace.path() if trace is not None else None
        callbacks: list[BaseCallbackHandler] = [trace] if trace is not None else []

        outcome = empty_outcome(trace_path = trace_path, timing = timing)

        if reuse_run_id is None:
            resume_point = None
            question, error = resolve_question(
                question_text = question_text,
                audio_path = audio_path,
                record_seconds = record_seconds,
                timing = timing,
            )
            if not error and not question.strip():
                error = "вопрос пустой"
        else:
            resume_point = find_resume_point(run_id = reuse_run_id, from_node = "compose")
            question = resume_point.question
            error = resume_point.error

        if error:
            yield failed_stage(outcome = replace(outcome, question = question), error = error)
            return

        outcome = replace(outcome, question = question)
        yield OmniStage(name = "вопрос", step = None, outcome = outcome)

        persona: Persona | None = None
        # Без нового рассказчика прогон на готовых фактах идёт с записанным.
        narrator_prompt = "" if resume_point is None else resume_point.narrator_prompt
        look = ""

        if narrator_style is not None:
            narrator_prompt = narrator_style
            log_narrator_style(style = narrator_prompt)
        elif image_path is not None:
            narrator_prompt, persona, look, error = build_narrator_from_image(
                image_path = image_path,
                persona_mode = persona_mode,
                timing = timing,
                callbacks = callbacks,
            )
            if error:
                yield failed_stage(outcome = replace(outcome, look = look), error = error)
                return

        if narrator_prompt:
            outcome = replace(
                outcome,
                persona = persona,
                narrator_prompt = narrator_prompt,
                look = look,
            )
            yield OmniStage(name = "рассказчик", step = None, outcome = outcome)

        if resume_point is None:
            research_steps = run_research_staged(
                question = question,
                narrator_prompt = narrator_prompt or None,
                run_id = run_id,
                callbacks = callbacks,
            )
        else:
            outcome = replace(outcome, notes = resume_point.notes)
            research_steps = resume_research_staged(
                point = resume_point,
                narrator_prompt = narrator_prompt or None,
                callbacks = callbacks,
            )

        with timing.stage(name = "ресёрч"):
            for step in research_steps:
                if step.notes is not None:
                    outcome = replace(outcome, notes = step.notes)
                if step.answer is not None:
                    outcome = replace(outcome, answer = step.answer)

                yield OmniStage(name = "ресёрч", step = step, outcome = outcome)

        if outcome.answer is None or outcome.notes is None:
            yield failed_stage(outcome = outcome, error = "граф не дал готового текста")
            return

        yield from finish_run_staged(
            outcome = outcome,
            is_speech_on = is_speech_on,
            is_markup_on = is_markup_on,
            callbacks = callbacks,
        )


def run_omni_assistant(
    image_path: Path | None,
    narrator_style: str | None,
    persona_mode: PersonaMode,
    question_text: str | None,
    audio_path: Path | None,
    record_seconds: float | None,
    reuse_run_id: str | None,
    is_speech_on: bool,
    is_markup_on: bool,
) -> OmniOutcome:
    """
    Проводит экскурсию по вопросу от лица заданного рассказчика.

    Промежуточные снимки прогона не нужны: возвращается только последний.

    Аргументы:
        image_path: файл с фотографией персонажа; None - без фотографии.
        narrator_style: готовая фраза про голос рассказчика; None - без неё.
        persona_mode: способ сборки рассказчика по фотографии.
        question_text: вопрос текстом; None - вопрос задан иначе.
        audio_path: файл с записью вопроса; None - вопрос задан иначе.
        record_seconds: сколько секунд писать с микрофона; RECORD_UNTIL_ENTER и
            меньше - до нажатия Enter; None - вопрос задан иначе.
        reuse_run_id: прогон, факты которого берутся готовыми; None - собирать
            факты заново.
        is_speech_on: озвучивать готовый текст.
        is_markup_on: размечать текст перед озвучкой.

    Возвращает:
        Исход прогона: текст экскурсии с опорой и озвучкой либо причина неудачи.
    """
    outcome: OmniOutcome | None = None

    for stage in run_omni_assistant_staged(
        image_path = image_path,
        narrator_style = narrator_style,
        persona_mode = persona_mode,
        question_text = question_text,
        audio_path = audio_path,
        record_seconds = record_seconds,
        reuse_run_id = reuse_run_id,
        is_speech_on = is_speech_on,
        is_markup_on = is_markup_on,
    ):
        outcome = stage.outcome

    return outcome


def resume_omni_assistant_staged(
    resume_run_id: str,
    from_node: str,
    narrator_style: str | None,
    is_speech_on: bool,
    is_markup_on: bool,
) -> Iterator[OmniStage]:
    """
    Переигрывает записанный прогон, отдавая снимок после каждого этапа.

    Вопрос и рассказчик берутся из снимка. Готовая фраза про рассказчика
    заменяет записанную. Персонажа полями в снимке нет, поэтому голос и разметка
    строятся по блоку про рассказчика.

    Аргументы:
        resume_run_id: идентификатор записанного прогона.
        from_node: узел, с которого продолжать.
        narrator_style: готовая фраза про голос рассказчика; None - взять
            записанную.
        is_speech_on: озвучивать готовый текст.
        is_markup_on: размечать текст перед озвучкой.

    Возвращает:
        Снимки прогона по одному, от продолженного текста до готовой озвучки.
        Оборвавшийся прогон отдаёт последним снимок с причиной неудачи.
    """
    timing = Stopwatch()

    # Журнал у продолжения свой: имя исходного прогона плюс время рестарта.
    # Исходный файл остаётся нетронутым, а происхождение видно и в имени, и в шапке.
    trace_id = f"{resume_run_id}+{new_run_id()}"
    origin_rows = [f"- продолжение прогона `{resume_run_id}` с узла `{from_node}`"]

    with trace_run(
        trace_id = trace_id,
        node_rows = describe_nodes(llms = build_node_llms()),
        origin_rows = origin_rows,
    ) as trace:
        trace_path = trace.path() if trace is not None else None
        callbacks: list[BaseCallbackHandler] = [trace] if trace is not None else []

        outcome = empty_outcome(trace_path = trace_path, timing = timing)

        with timing.stage(name = "продолжение"):
            resumed = resume_research(
                run_id = resume_run_id,
                from_node = from_node,
                narrator_prompt = narrator_style,
                callbacks = callbacks,
            )

        error = resumed.error
        if not error and (resumed.answer is None or resumed.notes is None):
            error = "продолжение не дало готового текста"

        if error:
            yield failed_stage(
                outcome = replace(outcome, question = resumed.question),
                error = error,
            )
            return

        outcome = replace(
            outcome,
            question = resumed.question,
            answer = resumed.answer,
            notes = resumed.notes,
            narrator_prompt = resumed.narrator_prompt,
        )

        yield from finish_run_staged(
            outcome = outcome,
            is_speech_on = is_speech_on,
            is_markup_on = is_markup_on,
            callbacks = callbacks,
        )


def resume_omni_assistant(
    resume_run_id: str,
    from_node: str,
    narrator_style: str | None,
    is_speech_on: bool,
    is_markup_on: bool,
) -> OmniOutcome:
    """
    Переигрывает записанный прогон с указанного узла.

    Промежуточные снимки прогона не нужны: возвращается только последний.

    Аргументы:
        resume_run_id: идентификатор записанного прогона.
        from_node: узел, с которого продолжать.
        narrator_style: готовая фраза про голос рассказчика; None - взять
            записанную.
        is_speech_on: озвучивать готовый текст.
        is_markup_on: размечать текст перед озвучкой.

    Возвращает:
        Исход прогона: текст экскурсии с опорой и озвучкой либо причина неудачи.
    """
    outcome: OmniOutcome | None = None

    for stage in resume_omni_assistant_staged(
        resume_run_id = resume_run_id,
        from_node = from_node,
        narrator_style = narrator_style,
        is_speech_on = is_speech_on,
        is_markup_on = is_markup_on,
    ):
        outcome = stage.outcome

    return outcome


def speak_outcome_staged(
    outcome: OmniOutcome,
    is_markup_on: bool,
    stamp: str,
    callbacks: list[BaseCallbackHandler],
) -> Iterator[OmniStage]:
    """
    Подбирает голос и озвучивает готовый текст, отдавая снимок после каждого события.

    Аргументы:
        outcome: исход прогона с готовым текстом; голос и файлы озвучки в нём
            заполняются по ходу.
        is_markup_on: размечать текст перед озвучкой.
        stamp: штамп времени прогона, общий с markdown прогона.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Снимки прогона по одному: сначала подобранный голос без файлов, дальше по
        снимку на каждое событие озвучки. Недоступный синтез даёт один снимок без
        голоса и без файлов.
    """
    synthesizer = SpeechSynthesizer(config = SPEAKING_CONFIG)

    with outcome.timing.stage(name = "загрузка синтеза"):
        speakers, error = synthesizer.available_speakers()
    if error:
        log_voices_missing(reason = error)
        yield OmniStage(name = "синтез недоступен", step = None, outcome = outcome)
        return

    writing_llm = build_llm(
        role = NodeRole.WRITING,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = None,
    )

    settings = default_voice(speakers = speakers)
    narrator_gender = UNKNOWN_GENDER

    if outcome.narrator_prompt:
        extraction_llm = build_llm(
            role = NodeRole.EXTRACTION,
            is_reasoning_forced = ENABLE_ALL_REASONING,
            model = None,
        )
        with outcome.timing.stage(name = "подбор голоса"):
            choice, error = pick_voice(
                llm = extraction_llm,
                narrator_prompt = outcome.narrator_prompt,
                speakers = speakers,
                speakers_by_gender = speakers_by_gender(),
                callbacks = callbacks,
            )
        if error:
            log_voice_fallback(reason = error)
        else:
            settings = choice.to_voice_settings()
            narrator_gender = choice.narrator_gender

    log_voice(settings = settings)
    outcome = replace(outcome, voice = settings)
    yield OmniStage(name = "голос", step = None, outcome = outcome)

    paths: list[Path] = []

    for name, path in speak_answer_staged(
        answer = outcome.answer,
        narrator_prompt = outcome.narrator_prompt,
        settings = settings,
        title_settings = title_voice(
            narrator_gender = narrator_gender,
            speakers = speakers,
        ),
        synthesizer = synthesizer,
        llm = writing_llm,
        is_markup_on = is_markup_on,
        stamp = stamp,
        timing = outcome.timing,
        callbacks = callbacks,
    ):
        if path is not None:
            paths.append(path)
            outcome = replace(outcome, audio_paths = list(paths))

        yield OmniStage(name = name, step = None, outcome = outcome)
