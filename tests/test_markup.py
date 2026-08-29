"""
Тесты санитайзера разметки ssml. Модели не запускаются.
"""

from assistant.integrations.speaking import sanitize_markup, wrap_speech_parts


def test_plain_text_stays_plain() -> None:
    """
    Проверяет, что текст без разметки проходит без тегов.
    """
    body, has_markup = sanitize_markup(text = "Здравствуйте, я ваш экскурсовод")

    assert body == "Здравствуйте, я ваш экскурсовод"
    assert has_markup is False


def test_allowed_tags_survive() -> None:
    """
    Проверяет, что теги белого списка остаются в теле.
    """
    body, has_markup = sanitize_markup(
        text = 'Тише<break time="500ms"/><prosody rate="fast">и быстрее</prosody>'
    )

    assert body == 'Тише<break time="500ms"/><prosody rate="fast">и быстрее</prosody>'
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


def test_wrong_attribute_value_drops_tag() -> None:
    """
    Проверяет, что prosody со значением вне схемы выбрасывается.
    """
    body, has_markup = sanitize_markup(text = '<prosody rate="turbo">Быстро</prosody>')

    assert body == "Быстро"
    assert has_markup is False


def test_bad_break_time_drops_tag() -> None:
    """
    Проверяет, что пауза без единицы измерения выбрасывается.
    """
    body, has_markup = sanitize_markup(text = 'Раз<break time="долго"/>два')

    assert body == "Раздва"
    assert has_markup is False


def test_unclosed_tag_gets_closed() -> None:
    """
    Проверяет, что незакрытый тег закрывается в конце тела.
    """
    body, has_markup = sanitize_markup(text = '<prosody pitch="low">Низко')

    assert body == '<prosody pitch="low">Низко</prosody>'
    assert has_markup is True


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


def test_tag_inside_word_is_dropped() -> None:
    """
    Проверяет, что пара тегов, разрезающая слово, выбрасывается вместе с парой.
    """
    body, has_markup = sanitize_markup(text = 'Этот <prosody rate="fast">з</prosody>ал')

    assert body == "Этот зал"
    assert has_markup is False


def test_tag_around_whole_word_survives() -> None:
    """
    Проверяет, что тег вокруг целого слова остаётся.
    """
    body, has_markup = sanitize_markup(text = 'Этот <prosody rate="fast">зал</prosody> помнит')

    assert body == 'Этот <prosody rate="fast">зал</prosody> помнит'
    assert has_markup is True


def test_tag_before_accent_mark_is_dropped() -> None:
    """
    Проверяет, что знак ударения считается частью слова.
    """
    body, has_markup = sanitize_markup(text = 'мол<prosody pitch="low">+от</prosody>')

    assert body == "мол+от"
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


def test_sentence_boundary_inside_tag_is_ignored() -> None:
    """
    Проверяет, что граница предложения внутри тега не делит текст.
    """
    body = '<prosody rate="slow">Раз. Два.</prosody>'

    assert wrap_speech_parts(body = body) == f"<p><s>{body}</s></p>"
