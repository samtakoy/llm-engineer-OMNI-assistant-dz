"""
Параметры сэмплирования по моделям и по характеру работы узла.

Слоя два. Профиль модели отвечает на вопрос «что любит эта модель»: каждая
хочет своих настроек, и выставлять их руками в ui сервера - способ однажды
забыть и получить необъяснимо плохой прогон. Перекрытие роли отвечает на вопрос
«что делает этот узел»: вызов инструмента и длинный художественный текст живут
на разных настройках одной и той же модели.

Накладываются они в build_llm: профиль модели, поверх него перекрытие роли,
поверх - разовые перекрытия окружения. Параметры уезжают в теле запроса и
перекрывают настройки сервера, поэтому этот модуль и есть источник истины.

Часть параметров вне стандарта OpenAI: temperature, top_p и presence_penalty -
обычные поля запроса, а top_k, min_p и repeat_penalty уходят в extra_body как
расширение lm studio. Провайдер, который их не знает, обычно молча игнорирует;
если начнёт отвечать 400, extra_body для него надо очистить.
"""

from dataclasses import dataclass, replace
from enum import Enum


class NodeRole(Enum):
    """
    Характер работы узла графа.

    Роль названа по работе, а не по имени узла: следующий узел с длинным
    текстом переиспользует WRITING, не заводя себе отдельной записи.
    """

    TOOL_CALLING = "tool_calling"
    EXTRACTION = "extraction"
    WRITING = "writing"


class Keep:
    """
    Маркер: поле роли не перекрывает профиль модели.

    Нужен отдельно от None, потому что None у параметров сэмплирования уже
    занят и означает «не отправлять параметр в запросе».
    """


KEEP = Keep()


@dataclass(frozen = True)
class RoleOverlay:
    """
    Перекрытие профиля модели под характер работы узла.

    Поля повторяют SamplingProfile. Значение KEEP оставляет значение профиля
    модели, None убирает параметр из запроса, число подставляет своё.

    Атрибуты:
        temperature: температура.
        top_p: ядерная выборка.
        presence_penalty: штраф за повтор темы.
        repeat_penalty: штраф за повтор токенов.
        reasoning_effort: бюджет размышления.
        max_tokens: потолок длины ответа.
        note: зачем роли эти значения.
    """

    temperature: float | None | Keep = KEEP
    top_p: float | None | Keep = KEEP
    presence_penalty: float | None | Keep = KEEP
    repeat_penalty: float | None | Keep = KEEP
    reasoning_effort: str | None | Keep = KEEP
    max_tokens: int | None | Keep = KEEP
    note: str = ""

    def overrides(self) -> dict[str, object]:
        """
        Возвращает перекрываемые поля вместе с маркерами KEEP.

        Разбирает их вызывающая сторона: здесь неизвестно, что стоит в профиле
        модели.

        Возвращает:
            Словарь: имя поля профиля - значение перекрытия либо KEEP.
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "repeat_penalty": self.repeat_penalty,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen = True)
class SamplingProfile:
    """
    Настройки сэмплирования одной модели.

    Атрибуты:
        temperature: температура; None - не отправлять.
        top_p: ядерная выборка.
        top_k: отсечение по числу кандидатов (extra_body).
        min_p: отсечение по доле вероятности лидера (extra_body).
        presence_penalty: штраф за повтор темы.
        repeat_penalty: штраф за повтор токенов (extra_body).
        reasoning_effort: бюджет размышления; None - не трогать.
        stop: последовательности остановки, если модель не останавливается сама.
        max_tokens: потолок длины ответа; None - без ограничения.
        note: откуда взяты значения.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None
    reasoning_effort: str | None = None
    stop: tuple[str, ...] | None = None
    max_tokens: int | None = None
    note: str = ""

    def standard(self) -> dict[str, object]:
        """
        Возвращает поля, которые понимает обычный openai-клиент.

        Возвращает:
            Словарь параметров запроса без None.
        """
        pairs: dict[str, object] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "stop": list(self.stop) if self.stop else None,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
        }
        return {key: value for key, value in pairs.items() if value is not None}

    def with_overlay(self, overlay: "RoleOverlay") -> "SamplingProfile":
        """
        Накладывает перекрытие роли на профиль модели.

        Аргументы:
            overlay: перекрытие роли узла.

        Возвращает:
            Новый профиль: поля со значением KEEP взяты из профиля модели,
            остальные - из перекрытия.
        """
        overridden = {
            name: getattr(self, name) if isinstance(value, Keep) else value
            for name, value in overlay.overrides().items()
        }

        notes = [note for note in (self.note, overlay.note) if note]
        return replace(self, **overridden, note = "; ".join(notes))

    def extra(self) -> dict[str, object]:
        """
        Возвращает параметры, живущие в extra_body.

        Здесь только то, чего нет в схеме openai: расширения llama.cpp, которые
        клиент иначе не пропустит.

        Возвращает:
            Словарь расширенных параметров без None.
        """
        pairs = {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repeat_penalty": self.repeat_penalty,
        }
        return {key: value for key, value in pairs.items() if value is not None}


# Ключ - подстрока имени модели, поиск по самому длинному совпадению.
PROFILES: dict[str, SamplingProfile] = {
    "gemma-4-26b-a4b": SamplingProfile(
        temperature = 1.0,
        top_p = 0.95,
        top_k = 64,
        note = "официальные параметры google",
    ),
    "glm-4.7-flash": SamplingProfile(
        temperature = 0.2,
        top_p = 0.95,
        top_k = 50,
        min_p = 0.01,
        reasoning_effort = "none",
        stop = ("<|user|>", "<|observation|>"),
        note = "model.yaml lmstudio-community; стоп-слова обязательны",
    ),
    "qwen3.6-35b-a3b": SamplingProfile(
        temperature = 1.0,
        top_p = 0.95,
        top_k = 20,
        min_p = 0.0,
        presence_penalty = 1.5,
        repeat_penalty = 1.0,
        reasoning_effort = "none",
        note = "рекомендованные параметры qwen; размышление выключено по умолчанию",
    ),
}

# Почему у qwen размышление выключено в профиле.
#
# Под грамматикой (структурированный вывод) оно ломает разбор: ответ целиком
# остаётся в reasoning_content, а content приходит пустым. Гасит это только
# значение "none" - low и minimal размышление лишь укорачивают.
#
# Без грамматики размышление работает и в content попадает нормальный ответ, но
# бюджет по умолчанию у qwen не ограничен: на вызовах инструментов модель уходит
# в рассуждение на десятки тысяч токенов и не останавливается. Поэтому роль
# TOOL_CALLING перекрывает профиль значением "low" и ставит потолок длины.
#
# Оговорка про presence_penalty = 1.5: значение агрессивное, его назначение -
# гасить повторы в длинных рассуждениях. На выводе оно толкает модель прочь от
# нужных слов, поэтому роли EXTRACTION и WRITING его снимают.

# Стоп-слова glm задаём руками: сборка объявляет три токена остановки
# (<|endoftext|>, <|user|>, <|observation|>), а tokenizer_config знает про один.
# Lm studio берёт второй файл, поэтому без stop в теле генерация не
# останавливается и теги попадают в текст. Настройки stop strings в ui на
# api-запросы не распространяются.

DEFAULT_PROFILE = SamplingProfile(temperature = 0.0, note = "нейтральный профиль")

# Перекрытия по характеру работы узла. Роль объявляет только то, что меняет:
# остальные поля остаются такими, какими их задал профиль модели.
ROLE_OVERLAYS: dict[NodeRole, RoleOverlay] = {
    NodeRole.TOOL_CALLING: RoleOverlay(
        reasoning_effort = "low",
        max_tokens = 9000,
        note = "штраф за повтор оставлен: гасит зацикливание на вызовах",
    ),
    NodeRole.EXTRACTION: RoleOverlay(
        temperature = 0.1,
        presence_penalty = None,
        reasoning_effort = "none",
        note = "грамматика: размышление оставляет content пустым",
    ),
    NodeRole.WRITING: RoleOverlay(
        temperature = 0.5,
        presence_penalty = None,
        repeat_penalty = 1.05,
        reasoning_effort = "none",
        note = "presence_penalty выбивал русские слова в соседний язык",
    ),
}

# Почему WRITING гасит presence_penalty.
#
# Штраф действует на каждый уже встретившийся токен. В длинном тексте под него
# попадают русские служебные слова, и модель ищет им замену - ближайшей
# оказывается словоформа соседнего славянского языка: то же значение, другой
# токен, штраф обнулён. Дальше срабатывает самоподкрепление, и текст целиком
# уезжает с языка.
#
# Повторы в длинном тексте гасить всё же надо, поэтому вместо presence_penalty
# роль ставит мягкий repeat_penalty: он работает по токенам, а не по присутствию
# темы, и не толкает модель прочь от нужных слов.
#
# Размышление выключено вынужденно: узел работает под грамматикой, а под ней
# ответ наружу не выходит - весь вывод остаётся в reasoning_content, content
# приходит пустым, и разбор структуры падает. Значения low и minimal не
# помогают, гасит только "none". Как получить размышление вместе со схемой -
# в docs/SO_with_reasoning.md.

# Потолок длины ответа стоит только у TOOL_CALLING: вызов инструмента короткий,
# финальная реплика тоже, а без потолка ушедшая в рассуждение модель крутится до
# упора. Узлам вывода потолок не нужен - там длину задаёт запрос.


def overlay_for(role: NodeRole) -> RoleOverlay:
    """
    Подбирает перекрытие по роли узла.

    Аргументы:
        role: характер работы узла.

    Возвращает:
        Перекрытие роли.
    """
    return ROLE_OVERLAYS[role]


def profile_for(model: str) -> SamplingProfile:
    """
    Подбирает профиль по имени модели.

    Аргументы:
        model: имя модели как его знает сервер.

    Возвращает:
        Профиль из реестра либо нейтральный.
    """
    key = model.lower()
    matches = [name for name in PROFILES if name in key]
    if not matches:
        return DEFAULT_PROFILE
    # Самое длинное совпадение точнее: «gemma-4-12b» не должен цеплять «gemma-4».
    return PROFILES[max(matches, key = len)]
