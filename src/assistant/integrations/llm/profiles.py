"""
Параметры сэмплирования: профиль модели, поверх него перекрытие роли узла.
Накладываются в build_llm и уезжают в теле запроса, перекрывая настройки сервера.
"""

from dataclasses import dataclass, field, fields, replace
from enum import Enum


class NodeRole(Enum):
    """
    Характер работы узла графа.

    Атрибуты:
        TOOL_CALLING: вызов инструментов.
        EXTRACTION: извлечение структуры.
        WRITING: длинный текст.
        VISION: разбор картинки.
    """

    TOOL_CALLING = "tool_calling"
    EXTRACTION = "extraction"
    WRITING = "writing"
    VISION = "vision"


class BodySection(Enum):
    """
    Секция тела запроса, куда уезжает параметр.

    Атрибуты:
        STANDARD: обычные поля запроса.
        EXTRA: поля extra_body.
    """

    STANDARD = "standard"
    EXTRA = "extra"


class Keep:
    """
    Маркер: поле роли не перекрывает профиль модели.
    """

    def __repr__(self) -> str:
        """
        Возвращает имя маркера.

        Возвращает:
            Строку KEEP.
        """
        return "KEEP"


KEEP = Keep()


@dataclass(frozen = True)
class RoleOverlay:
    """
    Перекрытие профиля модели под характер работы узла.

    Имена полей совпадают с именами полей SamplingProfile. KEEP оставляет
    значение профиля модели, None убирает параметр из запроса, число
    подставляет своё.

    Атрибуты:
        temperature: температура.
        top_p: ядерная выборка.
        presence_penalty: штраф за повтор темы.
        frequency_penalty: штраф за частоту повтора.
        repeat_penalty: штраф за повтор токенов.
        repeat_last_n: окно, по которому считаются штрафы.
        reasoning_effort: бюджет размышления.
        max_tokens: потолок длины ответа.
        note: зачем роли эти значения.
    """

    temperature: float | None | Keep = KEEP
    top_p: float | None | Keep = KEEP
    presence_penalty: float | None | Keep = KEEP
    frequency_penalty: float | None | Keep = KEEP
    repeat_penalty: float | None | Keep = KEEP
    repeat_last_n: int | None | Keep = KEEP
    reasoning_effort: str | None | Keep = KEEP
    max_tokens: int | None | Keep = KEEP
    note: str = ""


@dataclass(frozen = True)
class SamplingProfile:
    """
    Настройки сэмплирования одной модели.

    Атрибуты:
        temperature: температура; None - не отправлять.
        top_p: ядерная выборка.
        top_k: отсечение по числу кандидатов (extra_body).
        min_p: отсечение по доле вероятности лидера (extra_body).
        presence_penalty: штраф за повтор темы, одинаковый при любом числе повторов.
        frequency_penalty: штраф за повтор темы, растущий с числом повторов.
        repeat_penalty: штраф за повтор токенов (extra_body).
        repeat_last_n: сколько последних токенов видят штрафы; -1 - весь
            контекст, 0 - штрафы выключены (extra_body).
        reasoning_effort: бюджет размышления; None - не трогать.
        stop: последовательности остановки, если модель не останавливается сама.
        max_tokens: потолок длины ответа; None - без ограничения.
        note: откуда взяты значения.
        role_overlays: уточнение перекрытия роли под эту модель.
    """

    temperature: float | None = field(default = None, metadata = {"body": BodySection.STANDARD})
    top_p: float | None = field(default = None, metadata = {"body": BodySection.STANDARD})
    top_k: int | None = field(default = None, metadata = {"body": BodySection.EXTRA})
    min_p: float | None = field(default = None, metadata = {"body": BodySection.EXTRA})
    presence_penalty: float | None = field(default = None, metadata = {"body": BodySection.STANDARD})
    frequency_penalty: float | None = field(default = None, metadata = {"body": BodySection.STANDARD})
    repeat_penalty: float | None = field(default = None, metadata = {"body": BodySection.EXTRA})
    repeat_last_n: int | None = field(default = None, metadata = {"body": BodySection.EXTRA})
    reasoning_effort: str | None = field(default = None, metadata = {"body": BodySection.STANDARD})
    stop: tuple[str, ...] | None = field(default = None, metadata = {"body": BodySection.STANDARD})
    max_tokens: int | None = field(default = None, metadata = {"body": BodySection.STANDARD})
    note: str = ""
    role_overlays: dict[NodeRole, "RoleOverlay"] = field(default_factory = dict)

    def _body(self, section: BodySection) -> dict[str, object]:
        """
        Собирает поля одной секции тела запроса.

        Секция поля задана его разметкой. Поле без разметки в тело не попадает.

        Аргументы:
            section: секция тела запроса.

        Возвращает:
            Словарь параметров секции без None.
        """
        collected: dict[str, object] = {}

        for item in fields(self):
            if item.metadata.get("body") != section:
                continue

            value = getattr(self, item.name)
            if value is None:
                continue

            collected[item.name] = list(value) if isinstance(value, tuple) else value

        return collected

    def standard(self) -> dict[str, object]:
        """
        Возвращает поля, которые понимает обычный openai-клиент.

        Возвращает:
            Словарь параметров запроса без None.
        """
        return self._body(section = BodySection.STANDARD)

    def with_overlay(self, overlay: "RoleOverlay") -> "SamplingProfile":
        """
        Накладывает перекрытие роли на профиль модели.

        Аргументы:
            overlay: перекрытие роли узла.

        Возвращает:
            Новый профиль: поля со значением KEEP взяты из профиля модели,
            остальные - из перекрытия.
        """
        overridden: dict[str, object] = {}

        for item in fields(self):
            if not item.metadata.get("body"):
                continue

            value = getattr(overlay, item.name, KEEP)
            if not isinstance(value, Keep):
                overridden[item.name] = value

        notes = [note for note in (self.note, overlay.note) if note]
        return replace(self, **overridden, note = "; ".join(notes))

    def for_role(self, role: NodeRole) -> "SamplingProfile":
        """
        Собирает профиль узла: значения модели, поверх - роль, поверх - роль этой модели.

        Третий слой нужен, потому что общее перекрытие роли одно на все модели,
        а удачные для крупной модели значения ломают мелкую.

        Аргументы:
            role: характер работы узла.

        Возвращает:
            Профиль с наложенными перекрытиями.
        """
        tuned = self.with_overlay(overlay = overlay_for(role = role))
        model_overlay = self.role_overlays.get(role)

        if model_overlay is None:
            return tuned

        return tuned.with_overlay(overlay = model_overlay)

    def extra(self) -> dict[str, object]:
        """
        Возвращает параметры, живущие в extra_body.

        Возвращает:
            Словарь расширенных параметров без None.
        """
        return self._body(section = BodySection.EXTRA)


def _check_overlay_fields() -> None:
    """
    Проверяет, что каждое поле RoleOverlay имеет пару в SamplingProfile.

    Возвращает:
        Ничего. Поднимает RuntimeError, если пара не найдена.
    """
    profile_names = {item.name for item in fields(SamplingProfile)}
    unknown = sorted(item.name for item in fields(RoleOverlay) if item.name not in profile_names)

    if unknown:
        raise RuntimeError(
            f"Поля RoleOverlay без пары в SamplingProfile: {', '.join(unknown)}. "
            "Перекрытие такого поля потерялось бы молча."
        )


_check_overlay_fields()


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
    "qwen3.5-4b": SamplingProfile(
        temperature = 1.0,
        top_p = 0.95,
        top_k = 20,
        presence_penalty = 1.5,
        note = "параметры карточки модели на ollama",
    ),
    "qwen3.5-9b": SamplingProfile(
        temperature = 1.0,
        top_p = 0.95,
        top_k = 20,
        presence_penalty = 1.5,
        note = "параметры карточки модели на ollama",
    ),
    "qwen3.6-35b-a3b": SamplingProfile(
        temperature = 0.7,
        top_p = 0.80,
        top_k = 20,
        min_p = 0.0,
        presence_penalty = 1.5,
        repeat_penalty = 1.0,
        reasoning_effort = "none",
        note = "instruct-набор карточки qwen: профиль выключает размышление",
    ),
    "qwen3-vl-4b": SamplingProfile(
        temperature = 0.7,
        top_p = 0.80,
        top_k = 20,
        repeat_penalty = 1.0,
        role_overlays = {
            NodeRole.WRITING: RoleOverlay(
                temperature = 0.7,
                presence_penalty = 1.0,
                frequency_penalty = 0.4,
                repeat_penalty = 1.05,
                repeat_last_n = -1,
                note = "4b: узел зацикливался, а на окне по умолчанию повторял фразы между разделами",
            ),
            NodeRole.EXTRACTION: RoleOverlay(
                temperature = 0.7,
                repeat_penalty = 1.05,
                note = "4b: жадная выборка без единого штрафа за повтор",
            ),
            NodeRole.TOOL_CALLING: RoleOverlay(
                presence_penalty = None,
                repeat_penalty = 1.05,
                max_tokens = 2000,
                note = "4b: штраф за тему на длинном контексте давал петлю",
            ),
        },
        note = "generation_config.json из репозитория Qwen/Qwen3-VL-4B-Instruct",
    ),
}

DEFAULT_PROFILE = SamplingProfile(temperature = 0.0, note = "нейтральный профиль")

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
    NodeRole.VISION: RoleOverlay(
        reasoning_effort = "none",
        max_tokens = 600,
        note = "сэмплирование из карточки модели; потолок рубит цикл: описание кадра укладывается в 400 токенов",
    ),
}


def standard_field_names() -> tuple[str, ...]:
    """
    Возвращает имена полей, уезжающих в тело запроса обычными параметрами.

    Возвращает:
        Имена полей в порядке объявления.
    """
    return tuple(
        item.name
        for item in fields(SamplingProfile)
        if item.metadata.get("body") is BodySection.STANDARD
    )


def overlay_for(role: NodeRole) -> RoleOverlay:
    """
    Подбирает перекрытие по роли узла.

    Аргументы:
        role: характер работы узла.

    Возвращает:
        Перекрытие роли.
    """
    return ROLE_OVERLAYS[role]


# Знаки, которыми серверы разбивают имя модели на части. Одна модель у lm studio
# зовётся google/gemma-4-26b-a4b-qat, у ollama - gemma4:26b-a4b-it-q4_K_M.
# Совпадают они только после удаления этих знаков.
NAME_SEPARATORS = ("/", ":", "_", ".", "-", " ")


def normalize_model_name(model: str) -> str:
    """
    Приводит имя модели к виду, пригодному для поиска по реестру профилей.

    Аргументы:
        model: имя модели как его знает сервер.

    Возвращает:
        Имя в нижнем регистре без разделителей.
    """
    name = model.lower()

    for separator in NAME_SEPARATORS:
        name = name.replace(separator, "")

    return name


def profile_for(model: str) -> SamplingProfile:
    """
    Подбирает профиль по имени модели.

    Аргументы:
        model: имя модели как его знает сервер.

    Возвращает:
        Профиль из реестра либо нейтральный.
    """
    key = normalize_model_name(model = model)
    matches = [name for name in PROFILES if normalize_model_name(model = name) in key]
    if not matches:
        return DEFAULT_PROFILE
    return PROFILES[max(matches, key = len)]
