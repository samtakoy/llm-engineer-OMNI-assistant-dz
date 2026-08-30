"""
Тесты сборки промпта рассказчика. Модели не запускаются.
"""

from assistant.persona import Persona, render_narrator_prompt


def build_sample_persona() -> Persona:
    """
    Собирает персонажа для тестов.

    Возвращает:
        Персонажа с заполненными полями.
    """
    return Persona(
        name = "Смотритель маяка",
        gender = "мужской",
        character = "Ворчливый одиночка, привыкший говорить с морем громче, чем с людьми",
        address_to_listener = "salaga",
        speech_manner = "Говорит короткими фразами, с долгими паузами",
        favourite_words = ["шторм", "по курсу", "держись"],
        favourite_sounds = ["кхм", "у-у-у"],
        attitude_to_subject = "Меряет всё расстоянием до берега",
    )


def test_prompt_repeats_between_calls() -> None:
    """
    Проверяет, что одна персона даёт одинаковый текст при повторных вызовах.
    """
    persona = build_sample_persona()

    assert render_narrator_prompt(persona = persona) == render_narrator_prompt(persona = persona)


def test_prompt_contains_every_field() -> None:
    """
    Проверяет, что в текст попали все поля персонажа.
    """
    persona = build_sample_persona()
    prompt = render_narrator_prompt(persona = persona)

    assert persona.name in prompt
    assert persona.gender in prompt
    assert persona.character in prompt
    assert persona.address_to_listener in prompt
    assert persona.speech_manner in prompt
    assert persona.attitude_to_subject in prompt
    for word in persona.favourite_words:
        assert word in prompt
    for sound in persona.favourite_sounds:
        assert sound in prompt


def test_prompt_has_no_unfilled_placeholders() -> None:
    """
    Проверяет, что в тексте не осталось незаполненных мест шаблона.
    """
    prompt = render_narrator_prompt(persona = build_sample_persona())

    assert "{" not in prompt
    assert "}" not in prompt


def test_different_personas_give_different_prompts() -> None:
    """
    Проверяет, что разные персонажи дают разный текст.
    """
    first = build_sample_persona()
    second = first.model_copy(update = {"name": "Циркач", "address_to_listener": "почтенная публика"})

    assert render_narrator_prompt(persona = first) != render_narrator_prompt(persona = second)
