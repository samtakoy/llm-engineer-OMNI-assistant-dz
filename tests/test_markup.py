"""
Тесты санитайзера разметки ssml и резки текста на куски. Модели не запускаются.
"""

from assistant.integrations.speaking import sanitize_markup, split_into_chunks, wrap_speech_parts


def test_plain_text_stays_plain() -> None:
    """
    Проверяет, что текст без разметки проходит без тегов.
    """
    body, has_markup = sanitize_markup(text = "Здравствуйте, я ваш экскурсовод")

    assert body == "Здравствуйте, я ваш экскурсовод"
    assert has_markup is False


def test_allowed_tags_survive() -> None:
    """
    Проверяет, что пауза белого списка остаётся в теле.
    """
    body, has_markup = sanitize_markup(text = 'Тише<break time="500ms"/>и быстрее')

    assert body == 'Тише<break time="500ms"/>и быстрее'
    assert has_markup is True


def test_paragraph_and_sentence_tags_are_dropped() -> None:
    """
    Проверяет, что теги абзаца и предложения от модели не проходят: их ставит код.
    """
    body, has_markup = sanitize_markup(text = "<p><s>Речь</s></p>")

    assert body == "Речь"
    assert has_markup is False


def test_unknown_tag_becomes_nothing_and_text_stays() -> None:
    """
    Проверяет, что чужой тег выбрасывается, а его содержимое остаётся текстом.
    """
    body, has_markup = sanitize_markup(text = "<voice name='hulk'>Крушить</voice>")

    assert body == "Крушить"
    assert has_markup is False


def test_prosody_is_dropped() -> None:
    """
    Проверяет, что тег темпа и высоты выбрасывается: их задают настройки голоса.
    """
    body, has_markup = sanitize_markup(text = '<prosody rate="slow" pitch="low">Тише</prosody>')

    assert body == "Тише"
    assert has_markup is False


def test_bad_break_time_drops_tag() -> None:
    """
    Проверяет, что пауза без единицы измерения выбрасывается.
    """
    body, has_markup = sanitize_markup(text = 'Раз<break time="долго"/>два')

    assert body == "Раздва"
    assert has_markup is False


def test_unclosed_tag_is_dropped() -> None:
    """
    Проверяет, что незакрытый чужой тег выбрасывается, а текст остаётся.
    """
    body, has_markup = sanitize_markup(text = '<prosody pitch="low">Низко')

    assert body == "Низко"
    assert has_markup is False


def test_stray_closing_tag_is_dropped() -> None:
    """
    Проверяет, что закрывающий тег без открывающего выбрасывается.
    """
    body, has_markup = sanitize_markup(text = "Слово</prosody> и ещё")

    assert body == "Слово и ещё"
    assert has_markup is False


def test_special_characters_are_escaped() -> None:
    """
    Проверяет, что угловые скобки и амперсанд уезжают экранированными.
    """
    body, has_markup = sanitize_markup(text = "Маркс & Энгельс > все")

    assert body == "Маркс &amp; Энгельс &gt; все"
    assert has_markup is False


def test_accent_marks_pass_through() -> None:
    """
    Проверяет, что знак ударения остаётся в тексте.
    """
    body, has_markup = sanitize_markup(text = 'Прив+ет<break time="1s"/>др+уг')

    assert body == 'Прив+ет<break time="1s"/>др+уг'
    assert has_markup is True


def test_tag_inside_word_leaves_word_whole() -> None:
    """
    Проверяет, что чужой тег внутри слова не разрезает слово.
    """
    body, has_markup = sanitize_markup(text = 'Этот <prosody rate="fast">з</prosody>ал')

    assert body == "Этот зал"
    assert has_markup is False


def test_speak_tag_is_dropped() -> None:
    """
    Проверяет, что корневой тег от модели выбрасывается: его ставит код.
    """
    body, has_markup = sanitize_markup(text = "<speak>Речь</speak>")

    assert body == "Речь"
    assert has_markup is False


def test_sentences_get_wrapped() -> None:
    """
    Проверяет, что предложения абзаца расходятся по тегам s.
    """
    assert wrap_speech_parts(body = "Первое. Второе!") == "<p><s>Первое.</s><s>Второе!</s></p>"


def test_paragraphs_get_wrapped() -> None:
    """
    Проверяет, что пустая строка делит текст на абзацы.
    """
    assert wrap_speech_parts(body = "Первый.\n\nВторой.") == "<p><s>Первый.</s></p><p><s>Второй.</s></p>"


def test_pause_survives_sentence_wrapping() -> None:
    """
    Проверяет, что пауза между предложениями остаётся в теле.
    """
    body = 'Раз. <break time="500ms"/>Два.'
    expected = '<p><s>Раз.</s><s><break time="500ms"/>Два.</s></p>'

    assert wrap_speech_parts(body = body) == expected


def test_short_text_stays_one_chunk() -> None:
    """
    Проверяет, что текст короче бюджета уходит одним куском.
    """
    assert split_into_chunks(text = "Раз. Два. Три.", budget = 100) == ["Раз. Два. Три."]


def test_long_text_splits_by_sentences() -> None:
    """
    Проверяет, что текст длиннее бюджета режется по концу предложения.
    """
    text = f"{'а' * 30}. {'б' * 30}. {'в' * 30}."

    assert split_into_chunks(text = text, budget = 65) == [f"{'а' * 30}. {'б' * 30}.", f"{'в' * 30}."]


def test_pause_survives_splitting() -> None:
    """
    Проверяет, что пауза не разрезается и не тратит бюджет.
    """
    text = f"{'а' * 30}. <break time=\"500ms\"/>{'б' * 30}."

    assert split_into_chunks(text = text, budget = 65) == [text]


def test_long_sentence_splits_by_words() -> None:
    """
    Проверяет, что предложение длиннее бюджета дорезается по пробелам.
    """
    text = f"{'а' * 30} {'б' * 30} {'в' * 30}."

    assert split_into_chunks(text = text, budget = 65) == [f"{'а' * 30} {'б' * 30}", f"{'в' * 30}."]


def test_tags_survive_word_splitting() -> None:
    """
    Проверяет, что рез по пробелам не рассекает тег с атрибутом.
    """
    text = f"{'а' * 30} <break time=\"500ms\"/>{'б' * 30} {'в' * 30}"

    for chunk in split_into_chunks(text = text, budget = 65):
        assert chunk.count("<") == chunk.count(">")


def test_long_word_stays_whole() -> None:
    """
    Проверяет, что слово длиннее бюджета остаётся одним куском: резать нечего.
    """
    assert split_into_chunks(text = "а" * 50, budget = 10) == ["а" * 50]
