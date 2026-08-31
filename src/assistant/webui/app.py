"""
Веб-точка входа: фотография персонажа, вопрос текстом или голосом, экскурсия
несколькими проигрывателями.

Слой тонкий: работу ведёт assistant.omni, здесь только разметка и перекладка
снимков прогона в поля интерфейса.
"""

from collections.abc import Iterator
from pathlib import Path

import gradio as gr

from ..graph_runs import latest_run_id
from ..observability import setup_console_output
from ..omni import OmniOutcome, empty_outcome, run_omni_assistant_staged
from ..persona import PersonaMode
from ..timing import Stopwatch
from ..variables import PERSONA_MODE, SPOKEN_PATH
from .render import (
    render_answer,
    render_notes,
    render_persona,
    render_progress,
    render_timing,
    stage_lines,
)

# Сколько проигрывателей держать наготове. Куски ответа - вступление, разделы и
# завершение, - и их число заранее неизвестно, поэтому проигрыватели создаются
# заранее скрытыми и открываются по мере готовности файлов.
AUDIO_PLAYER_COUNT = 12


def resolve_narrator(
    image_path: str | None,
    narrator_style: str | None,
) -> tuple[Path | None, str | None]:
    """
    Выбирает источник рассказчика: фразу либо фотографию.

    Непустая фраза перебивает фотографию.

    Аргументы:
        image_path: файл с фотографией персонажа; None - фотографии нет.
        narrator_style: фраза про рассказчика; None либо пустая строка - фразы
            нет: нетронутое поле ввода приходит из gradio значением None.

    Возвращает:
        Пару «фотография, фраза». Незанятый источник равен None.
    """
    if narrator_style and narrator_style.strip():
        return None, narrator_style.strip()

    if image_path:
        return Path(image_path), None

    return None, None


def resolve_question_source(
    question_text: str | None,
    question_audio_path: str | None,
) -> tuple[str | None, Path | None]:
    """
    Выбирает источник вопроса: текст либо запись.

    Непустой текст перебивает запись.

    Аргументы:
        question_text: вопрос текстом; None либо пустая строка - текста нет:
            нетронутое поле ввода приходит из gradio значением None.
        question_audio_path: файл с записью вопроса; None - записи нет.

    Возвращает:
        Пару «текст, файл с записью». Незанятый источник равен None.
    """
    if question_text and question_text.strip():
        return question_text.strip(), None

    if question_audio_path:
        return None, Path(question_audio_path)

    return None, None


def player_updates(audio_paths: list[Path]) -> list[dict]:
    """
    Собирает обновления проигрывателей под готовые файлы озвучки.

    Аргументы:
        audio_paths: файлы с озвучкой в порядке произнесения.

    Возвращает:
        Обновление на каждый проигрыватель пула: занятые показываются с файлом,
        остальные прячутся. Файлы сверх пула не показываются.
    """
    updates: list[dict] = []

    for index in range(AUDIO_PLAYER_COUNT):
        if index < len(audio_paths):
            updates.append(
                gr.update(
                    value = str(audio_paths[index]),
                    visible = True,
                    label = f"кусок {index + 1}",
                )
            )
        else:
            updates.append(gr.update(value = None, visible = False))

    return updates


def render_status(stage_name: str, outcome: OmniOutcome) -> str:
    """
    Собирает строку состояния прогона.

    Аргументы:
        stage_name: имя завершённого этапа.
        outcome: исход прогона на этот момент.

    Возвращает:
        Строку с этапом либо с причиной неудачи, с припиской про непоказанные
        куски озвучки.
    """
    if outcome.error:
        return f"**прогон не удался:** {outcome.error}"

    line = f"**этап:** {stage_name}"

    hidden_count = len(outcome.audio_paths) - AUDIO_PLAYER_COUNT
    if hidden_count > 0:
        line = f"{line} (кусков озвучки больше, чем проигрывателей: {hidden_count} не показаны)"

    return line


def interface_fields(status: str, outcome: OmniOutcome, progress_lines: list[str]) -> tuple:
    """
    Раскладывает снимок прогона по полям интерфейса.

    Единственное место, где собирается набор выходов: их порядок совпадает со
    списком outputs у кнопки.

    Аргументы:
        status: строка состояния.
        outcome: исход прогона на этот момент.
        progress_lines: пройденные этапы в порядке прохождения.

    Возвращает:
        Значения полей в порядке выходов обработчика: состояние, ход прогона,
        персонаж, текст лекции, опора, длительности и обновления проигрывателей.
    """
    return (
        status,
        render_progress(lines = progress_lines),
        render_persona(
            look = outcome.look,
            persona = outcome.persona,
            narrator_prompt = outcome.narrator_prompt,
        ),
        render_answer(answer = outcome.answer),
        render_notes(notes = outcome.notes),
        render_timing(timing = outcome.timing),
        *player_updates(audio_paths = outcome.audio_paths),
    )


def blank_fields(status: str) -> tuple:
    """
    Собирает поля интерфейса для прогона, ещё не давшего ни одного снимка.

    Аргументы:
        status: строка состояния.

    Возвращает:
        Значения полей в порядке выходов обработчика.
    """
    return interface_fields(
        status = status,
        outcome = empty_outcome(trace_path = None, timing = Stopwatch()),
        progress_lines = [],
    )


def run_from_ui(
    image_path: str | None,
    question_text: str | None,
    question_audio_path: str | None,
    narrator_style: str | None,
    persona_mode_value: str,
    is_reuse_facts: bool,
    is_speech_on: bool,
    is_markup_on: bool,
) -> Iterator[tuple]:
    """
    Ведёт прогон по нажатию кнопки и отдаёт поля интерфейса после каждого этапа.

    Аргументы:
        image_path: файл с фотографией персонажа; None - фотографии нет.
        question_text: вопрос текстом; None - поле не тронуто.
        question_audio_path: файл с записью вопроса; None - записи нет.
        narrator_style: фраза про рассказчика; None - поле не тронуто.
        persona_mode_value: способ сборки рассказчика по фотографии.
        is_reuse_facts: взять факты свежего записанного прогона, сбор пропустить.
        is_speech_on: озвучивать готовый текст.
        is_markup_on: размечать текст перед озвучкой.

    Возвращает:
        Значения полей интерфейса по одному набору на этап прогона.
    """
    reuse_run_id = latest_run_id() if is_reuse_facts else ""
    if is_reuse_facts and not reuse_run_id:
        yield blank_fields(status = "**записанных прогонов нет:** факты брать неоткуда")
        return

    question, audio_path = resolve_question_source(
        question_text = question_text,
        question_audio_path = question_audio_path,
    )
    # С готовыми фактами вопрос берётся из записанного прогона: они собраны под него.
    if not is_reuse_facts and question is None and audio_path is None:
        yield blank_fields(status = "**нужен вопрос:** текстом или записью")
        return

    yield blank_fields(status = "**этап:** прогон начат")

    progress_lines: list[str] = []

    narrator_image, narrator_phrase = resolve_narrator(
        image_path = image_path,
        narrator_style = narrator_style,
    )

    for stage in run_omni_assistant_staged(
        image_path = narrator_image,
        narrator_style = narrator_phrase,
        persona_mode = PersonaMode(persona_mode_value),
        question_text = question,
        audio_path = audio_path,
        record_seconds = None,
        reuse_run_id = reuse_run_id or None,
        is_speech_on = is_speech_on,
        is_markup_on = is_markup_on,
    ):
        progress_lines.extend(stage_lines(name = stage.name, step = stage.step))
        yield interface_fields(
            status = render_status(stage_name = stage.name, outcome = stage.outcome),
            outcome = stage.outcome,
            progress_lines = progress_lines,
        )


def build_app() -> gr.Blocks:
    """
    Собирает интерфейс приложения.

    Возвращает:
        Готовый блок gradio, ещё не запущенный.
    """
    setup_console_output()

    with gr.Blocks(title = "Омни-ассистент") as app:
        gr.Markdown("# Экскурсия голосом персонажа с фотографии")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label = "Фотография персонажа", type = "filepath")
                question_input = gr.Textbox(label = "Вопрос текстом", lines = 3)
                question_audio_input = gr.Audio(
                    label = "Вопрос голосом: надиктовать или загрузить файл",
                    sources = ["microphone", "upload"],
                    type = "filepath",
                )

                with gr.Accordion(label = "Настройки прогона", open = False):
                    persona_mode_input = gr.Radio(
                        label = "Сборка рассказчика по фотографии",
                        choices = [mode.value for mode in PersonaMode],
                        value = PERSONA_MODE,
                    )
                    narrator_input = gr.Textbox(
                        label = "Рассказчик фразой; перебивает фотографию",
                        lines = 2,
                    )
                    reuse_facts_input = gr.Checkbox(
                        label = "Взять факты последнего прогона; вопрос берётся оттуда же",
                        value = False,
                    )
                    speech_input = gr.Checkbox(label = "Озвучивать текст", value = True)
                    markup_input = gr.Checkbox(label = "Размечать текст перед озвучкой", value = True)

                run_button = gr.Button(value = "Провести экскурсию", variant = "primary")

            with gr.Column():
                status_output = gr.Markdown(label = "Состояние")

                with gr.Accordion(label = "Ход прогона", open = True):
                    progress_output = gr.Markdown()

                with gr.Accordion(label = "Персонаж", open = True):
                    persona_output = gr.Markdown()

                answer_output = gr.Markdown(label = "Текст лекции")

                with gr.Accordion(label = "Опора и источники", open = False):
                    notes_output = gr.Markdown()

                with gr.Accordion(label = "Длительности этапов", open = False):
                    timing_output = gr.Markdown()

                gr.Markdown("### Озвучка")
                audio_outputs = [
                    gr.Audio(label = f"кусок {index + 1}", visible = False)
                    for index in range(AUDIO_PLAYER_COUNT)
                ]

        # Прогон один за раз: два параллельных подрались бы за модели
        # распознавания и синтеза.
        run_button.click(
            fn = run_from_ui,
            inputs = [
                image_input,
                question_input,
                question_audio_input,
                narrator_input,
                persona_mode_input,
                reuse_facts_input,
                speech_input,
                markup_input,
            ],
            outputs = [
                status_output,
                progress_output,
                persona_output,
                answer_output,
                notes_output,
                timing_output,
                *audio_outputs,
            ],
            concurrency_limit = 1,
        )

    return app


def launch_app(is_inline: bool, is_shared: bool) -> None:
    """
    Поднимает приложение.

    Аргументы:
        is_inline: встроить интерфейс в клетку ноутбука.
        is_shared: выдать публичную ссылку. Нужна в colab: во встроенном фрейме
            микрофон работает не всегда.

    Возвращает:
        Ничего.
    """
    # Без allowed_paths gradio не отдаёт файлы озвучки: они лежат вне каталога,
    # который он считает своим.
    build_app().launch(
        inline = is_inline,
        share = is_shared,
        allowed_paths = [str(SPOKEN_PATH)],
    )


def main() -> None:
    """
    Поднимает приложение из командной строки.

    Возвращает:
        Ничего.
    """
    launch_app(is_inline = False, is_shared = False)


if __name__ == "__main__":
    main()
