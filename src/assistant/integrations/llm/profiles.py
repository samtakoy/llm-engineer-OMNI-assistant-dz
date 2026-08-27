"""
Параметры сэмплирования по моделям.

Каждая модель хочет своих настроек, и выставлять их руками в UI сервера - способ
однажды забыть и получить необъяснимо плохой прогон. Параметры в теле запроса
перекрывают настройки сервера, поэтому реестр здесь и есть источник истины.

Часть параметров вне стандарта OpenAI: temperature, top_p и presence_penalty -
обычные поля запроса, а top_k, min_p и repeat_penalty уходят в extra_body как
расширение lm studio. Провайдер, который их не знает, обычно молча игнорирует;
если начнёт отвечать 400, extra_body для него надо очистить.
"""

from dataclasses import dataclass


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
        }
        return {key: value for key, value in pairs.items() if value is not None}

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
# в рассуждение на десятки тысяч токенов и не останавливается. Поэтому узлы,
# которым размышление нужно видимым, перекрывают профиль значением "low".
#
# Оговорка про presence_penalty = 1.5: значение агрессивное, его назначение -
# гасить повторы в длинных рассуждениях. На коротком структурированном выводе
# штраф может толкать модель прочь от нужных слов. Если ответ поедет - первый
# подозреваемый.

# Стоп-слова glm задаём руками: сборка объявляет три токена остановки
# (<|endoftext|>, <|user|>, <|observation|>), а tokenizer_config знает про один.
# Lm studio берёт второй файл, поэтому без stop в теле генерация не
# останавливается и теги попадают в текст. Настройки stop strings в ui на
# api-запросы не распространяются.

DEFAULT_PROFILE = SamplingProfile(temperature = 0.0, note = "нейтральный профиль")


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
