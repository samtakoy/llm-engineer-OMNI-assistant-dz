"""
Ручная проверка цепочки персонажа: картинка - облик - рассказчик.

Прогон идёт в два прохода: сначала все картинки разбирает vl-модель, потом по
готовым описаниям текстовая модель строит персонажей. Так модели меняются в
памяти один раз, а не на каждой картинке.

Запуск:
    .venv/bin/python scripts/check_persona.py docs/persona_images/hulk/images4.jpeg

```
# один персонаж целиком
.venv/bin/python scripts/check_persona.py docs/persona_images/hulk

# первые 10 из всех 50
.venv/bin/python scripts/check_persona.py docs/persona_images --limit 10

# по одной картинке из каждой папки
for folder in docs/persona_images/*/; do .venv/bin/python scripts/check_persona.py "$folder" --limit 1; done

```

"""

import argparse
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.observability import setup_console_output
from assistant.persona import Persona, build_persona, describe_look
from assistant.variables import VISION_MODEL, VISION_PROVIDER

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def collect_images(path: Path, limit: int) -> list[Path]:
    """
    Собирает список картинок по пути.

    Аргументы:
        path: файл с картинкой или каталог, который обходится вглубь.
        limit: сколько взять; ноль и меньше - все.

    Возвращает:
        Пути к картинкам по возрастанию имени.
    """
    if path.is_file():
        found = [path]
    else:
        found = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )

    return found[:limit] if limit > 0 else found


def collect_looks(llm: ChatOpenAI, images: list[Path]) -> dict[Path, str]:
    """
    Разбирает облик на каждой картинке и печатает результат.

    Аргументы:
        llm: клиент модели, принимающей картинки.
        images: пути к картинкам.

    Возвращает:
        Описания облика по путям к картинкам. Неудачные картинки в словарь
        не попадают.
    """
    looks: dict[Path, str] = {}

    for image_path in images:
        started = time.monotonic()
        look, error = describe_look(llm = llm, image_path = image_path, callbacks = [])
        spent_seconds = time.monotonic() - started

        if error:
            print(f"{image_path}: {error}, {spent_seconds:.1f} с\n", flush = True)
            continue

        print(f"{image_path}: облик за {spent_seconds:.1f} с", flush = True)
        print(f"{look.strip()}\n", flush = True)
        looks[image_path] = look

    return looks


def print_persona(persona: Persona) -> None:
    """
    Печатает поля рассказчика по одному в строке.

    Аргументы:
        persona: рассказчик, выведенный из облика.

    Возвращает:
        Ничего.
    """
    print(f"    имя:        {persona.name}", flush = True)
    print(f"    пол:        {persona.gender}", flush = True)
    print(f"    характер:   {persona.character}", flush = True)
    print(f"    обращение:  {persona.address_to_listener}", flush = True)
    print(f"    манера:     {persona.speech_manner}", flush = True)
    print(f"    словечки:   {', '.join(persona.favourite_words)}", flush = True)
    print(f"    звуки:   {', '.join(persona.favourite_sounds)}", flush = True)
    print(f"    отношение:  {persona.attitude_to_subject}", flush = True)


def build_personas(llm: ChatOpenAI, looks: dict[Path, str]) -> None:
    """
    Строит рассказчика по каждому описанию облика и печатает результат.

    Аргументы:
        llm: клиент текстовой модели.
        looks: описания облика по путям к картинкам.

    Возвращает:
        Ничего.
    """
    for image_path, look in looks.items():
        started = time.monotonic()
        persona, error = build_persona(llm = llm, look = look, callbacks = [])
        spent_seconds = time.monotonic() - started

        if error:
            print(f"{image_path}: {error}, {spent_seconds:.1f} с\n", flush = True)
            continue

        print(f"{image_path}: персонаж за {spent_seconds:.1f} с", flush = True)
        print_persona(persona = persona)
        print(flush = True)


def main() -> None:
    """
    Прогоняет картинки по указанному пути через зрение и сборку персонажа.

    Возвращает:
        Ничего.
    """
    setup_console_output()
    parser = argparse.ArgumentParser(description = "Проверка цепочки облик - персонаж")
    parser.add_argument("path", help = "файл с картинкой или каталог с картинками")
    parser.add_argument("--limit", type = int, default = 0, help = "сколько взять; ноль - все")
    arguments = parser.parse_args()

    images = collect_images(path = Path(arguments.path), limit = arguments.limit)

    vision_llm = build_llm(
        role = NodeRole.VISION,
        is_reasoning_forced = False,
        model = VISION_MODEL,
        provider = VISION_PROVIDER,
    )
    print(f"--- облик: [{VISION_PROVIDER}] {VISION_MODEL}, картинок {len(images)}\n", flush = True)
    looks = collect_looks(llm = vision_llm, images = images)

    writing_llm = build_llm(role = NodeRole.WRITING, is_reasoning_forced = False, model = None)
    print(f"--- персонаж: {writing_llm.model_name}, описаний {len(looks)}\n", flush = True)
    build_personas(llm = writing_llm, looks = looks)


if __name__ == "__main__":
    main()
