"""
Ручная проверка сервиса зрения: картинка уходит в модель, ответ приходит текстом.

Запуск:
    .venv/bin/python scripts/check_vision.py docs/persona_images/hulk/images4.jpeg

```
# один персонаж целиком
.venv/bin/python scripts/check_vision.py docs/persona_images/hulk

# первые 10 из всех 50
.venv/bin/python scripts/check_vision.py docs/persona_images --limit 10

# все 50
.venv/bin/python scripts/check_vision.py docs/persona_images

# Для разнообразия персонажей лучше по одной из каждой папки:
for folder in docs/persona_images/*/; do .venv/bin/python scripts/check_vision.py "$folder" --limit 1; done

```

"""

import argparse
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.integrations.llm.vision import describe_image, image_data_url
from assistant.observability import setup_console_output
from assistant.variables import (
    VISION_JPEG_QUALITY,
    VISION_MAX_SIDE,
    VISION_MODEL,
    VISION_PROVIDER,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")

LOOK_PROMPT = """\
Ты описываешь картинку для художника, который нарисует этого персонажа заново,
не видя оригинала.

Задача: перечислить видимое в кадре, назвать настроение кадра и, если узнаёшь
персонажа, назвать его. Это нужно, чтобы следующий этап придумал по описанию
манеру речи рассказчика.

Алгоритм работы:
1. Найди в кадре главную фигуру. Если фигуры нет, описывай сам кадр.
2. Опиши облик: телосложение, лицо, волосы, кожа, приметы.
3. Опиши текущую эмоцию персонажа.
4. Опиши позу, жест и выражение лица.
5. Назови настроение кадра: что чувствует смотрящий.
6. Опиши одежду и предметы в руках.
7. Опиши обстановку вокруг, свет и цвета.
8. Последней строкой напиши «Кто это: » и имя персонажа на английском языке вместе с тем, откуда он
   взят. Но если не уверен / не узнаешь, ТО напиши «Похож на: » и название персонажа.

Правила:
- 3-6 абзацев в порядке алгоритма, следом строка «Кто это: ».
- Абзац - одно-два предложения.
- Каждое утверждение об облике проверяется по кадру.
- Эмоцию называй вместе с её признаком: брови сведены, широкая улыбка и т.д..
- Надписи в кадре бери как подсказку для узнавания.
- Пиши по-русски простыми предложениями.

Запрещено:
- Называть персонажа наугад: сомневаешься - «не узнаю».
- Пересказывать подписи художников, водяные знаки и адреса сайтов.
- Повторять написанное другими словами.
"""


# Промпт предыдущего варианта.

# Опиши, что видно на картинке. Отвечай только по видимому.
# Если персонаж известный - назови его.
# Опиши также общий характер и эмоциональный фон картинки.
# Если бы он заговорил, то какую речь ожидать от него (например, немногословен, оживленный, хриплый, звонкий, депрессивный, радостный, ровный, приятный и т.д.)?
# После ответа добавь резюме: кто это 1-3 словами.


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


def check_image(llm: ChatOpenAI, image_path: Path) -> None:
    """
    Прогоняет одну картинку через сервис зрения и печатает результат.

    Аргументы:
        llm: клиент модели, принимающей картинки.
        image_path: файл с картинкой.

    Возвращает:
        Ничего.
    """
    started = time.monotonic()

    image_url, error = image_data_url(
        image_path = image_path,
        max_side = VISION_MAX_SIDE,
        jpeg_quality = VISION_JPEG_QUALITY,
    )
    if error:
        print(f"{image_path}: {error}", flush = True)
        return

    answer, error = describe_image(
        llm = llm,
        image_url = image_url,
        instruction = LOOK_PROMPT,
        callbacks = [],
    )
    spent_seconds = time.monotonic() - started

    if error:
        print(f"{image_path}: {error}, {spent_seconds:.1f} с", flush = True)
        return

    print(f"{image_path}: {len(image_url) // 1024} КБ, {spent_seconds:.1f} с", flush = True)
    print(f"    {answer}\n", flush = True)


def main() -> None:
    """
    Прогоняет картинки по указанному пути через сервис зрения.

    Возвращает:
        Ничего.
    """
    setup_console_output()
    parser = argparse.ArgumentParser(description = "Проверка сервиса зрения на картинках")
    parser.add_argument("path", help = "файл с картинкой или каталог с картинками")
    parser.add_argument("--limit", type = int, default = 0, help = "сколько взять; ноль - все")
    arguments = parser.parse_args()

    images = collect_images(path = Path(arguments.path), limit = arguments.limit)
    llm = build_llm(
        role = NodeRole.VISION,
        is_reasoning_forced = False,
        model = VISION_MODEL,
        provider = VISION_PROVIDER,
    )

    print(f"[{VISION_PROVIDER}] {VISION_MODEL}, картинок {len(images)}\n")
    for image_path in images:
        check_image(llm = llm, image_path = image_path)


if __name__ == "__main__":
    main()
