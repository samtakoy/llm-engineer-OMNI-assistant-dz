"""
Ручная проверка цепочки аватарки: картинка - облик - тег-промпт - рисунок.

Прогон идёт в три прохода: сначала все картинки разбирает vl-модель, потом
текстовая модель переводит описания в английские тег-промпты, и только затем
поднимается локальный Stable Diffusion. Так тяжёлые веса грузятся один раз, а
сетевые модели не ждут в памяти, пока считается картинка.

Нужны дополнительные пакеты:
    uv add "torch>=2.5,<3.0" "diffusers>=0.31,<1.0" "transformers>=4.44,<5.0" \
           "accelerate>=1.0,<2.0" "peft>=0.13,<1.0"

Запуск:
    .venv/bin/python scripts/check_avatar.py docs/persona_images/yoda/images.jpeg

```
# Батч
.venv/bin/python scripts/check_avatar.py docs/persona_images --mode quality


# один персонаж целиком
.venv/bin/python scripts/check_avatar.py docs/persona_images/yoda

# по одной картинке из каждой папки, быстрый режим
for folder in docs/persona_images/*/; do .venv/bin/python scripts/check_avatar.py "$folder" --limit 1; done

# медленный режим: полный график шумоподавления вместо четырёх шагов
.venv/bin/python scripts/check_avatar.py docs/persona_images/yoda --mode quality
for folder in docs/persona_images/*/; do .venv/bin/python scripts/check_avatar.py "$folder" --limit 1; done

# разброс вариантов на одном персонаже
for seed in 1 2 3 4; do .venv/bin/python scripts/check_avatar.py docs/persona_images/yoda --seed $seed; done

```

"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionPipeline
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, Field

from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.observability import setup_console_output
from assistant.persona import describe_look
from assistant.variables import PROJECT_ROOT, VISION_MODEL, VISION_PROVIDER

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")

# Веса на диске. Исходный репозиторий runwayml снят с hugging face, рабочее
# зеркало лежит под именем самой модели.
DIFFUSION_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Ускоряющий адаптер: четыре шага вместо двадцати пяти за 130 мегабайт весов.
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

# Настройки двух режимов: число шагов и сила следования промпту. У lcm сила
# равна единице - адаптер обучен без classifier-free guidance.
FAST_STEPS = 4
FAST_GUIDANCE = 1.0
QUALITY_STEPS = 25
QUALITY_GUIDANCE = 7.5

# Родное разрешение sd 1.5. Выше идут дубли лиц и вторые головы.
IMAGE_SIDE = 512

# Потолок текстового энкодера clip. Хвост промпта за этой границей молча
# отбрасывается, поэтому длина проверяется до вызова.
PROMPT_TOKEN_LIMIT = 77

AVATAR_PROMPT = """\
Ты собираешь промпт для Stable Diffusion 1.5 по готовому описанию персонажа.

Задача: превратить русское описание облика в короткий английский тег-промпт.
Это нужно, чтобы генератор нарисовал аватарку рассказчика экскурсии.

Алгоритм работы:
1. Найди в описании строку «Кто это: » или «Похож на: » и возьми оттуда имя
   персонажа вместе с источником.
2. Поставь это имя первыми тегами промпта, если персонаж назван уверенно.
3. Добавь теги облика: цвет и состояние кожи, волосы, приметы лица, возраст.
4. Добавь теги одежды и предметов в руках.
5. Добавь теги позы и выражения лица.
6. Добавь теги света, фона и кадрирования.

Правила:
- Только английский язык.
- Теги через запятую, без предложений и без глаголов в личной форме.
- Не длиннее 55 слов: текстовый энкодер режет промпт на 77 токенах.
- Порядок тегов = порядок важности, имя персонажа идёт первым.
- Кадрирование поясное или портретное: аватарке нужно лицо, а не полный рост.

Запрещено:
- Русские слова в ответе.
- Придумывать детали, которых нет в описании.
- Вес тегов в скобках вида (word:1.2): голый diffusers такой синтаксис не читает.
- Перечислять больше двух предметов в руках.
"""


class AvatarPrompt(BaseModel):
    """
    Готовый тег-промпт для генератора.

    Атрибуты:
        prompt: строка английских тегов через запятую.
    """

    prompt: str = Field(description = "английские теги через запятую")


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
        looks[image_path] = look

    return looks


def build_avatar_prompt(llm: ChatOpenAI, look: str) -> tuple[str, str]:
    """
    Переводит описание облика в английский тег-промпт.

    Аргументы:
        llm: клиент текстовой модели.
        look: описание облика персонажа на русском.

    Возвращает:
        Пару «тег-промпт, причина неудачи». При успехе причина пустая, при
        неудаче промпт пустой.
    """
    if not look.strip():
        return "", "описание облика пустое"

    structured_llm = llm.with_structured_output(AvatarPrompt, method = "json_schema")

    try:
        answer = structured_llm.invoke(
            [
                {"role": "system", "content": AVATAR_PROMPT},
                {"role": "user", "content": f"Описание облика:\n{look}"},
            ]
        )
    except Exception as error:
        print(f"[аватарка] вызов модели не удался: {type(error).__name__}: {error}")
        return "", f"модель не ответила по схеме: {type(error).__name__}"

    return answer.prompt.strip(), ""


def collect_prompts(llm: ChatOpenAI, looks: dict[Path, str]) -> dict[Path, str]:
    """
    Собирает тег-промпт по каждому описанию облика и печатает результат.

    Аргументы:
        llm: клиент текстовой модели.
        looks: описания облика по путям к картинкам.

    Возвращает:
        Тег-промпты по путям к картинкам. Неудачные описания в словарь не
        попадают.
    """
    prompts: dict[Path, str] = {}

    for image_path, look in looks.items():
        started = time.monotonic()
        prompt, error = build_avatar_prompt(llm = llm, look = look)
        spent_seconds = time.monotonic() - started

        if error:
            print(f"{image_path}: {error}, {spent_seconds:.1f} с\n", flush = True)
            continue

        print(f"{image_path}: промпт за {spent_seconds:.1f} с", flush = True)
        print(f"    {prompt}\n", flush = True)
        prompts[image_path] = prompt

    return prompts


def pick_device() -> tuple[str, torch.dtype]:
    """
    Выбирает устройство расчёта и тип чисел под него.

    Возвращает:
        Пару «имя устройства, тип чисел». На процессоре половинная точность
        не поддерживается, поэтому там остаётся float32.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16

    if torch.backends.mps.is_available():
        return "mps", torch.float16

    return "cpu", torch.float32


def build_pipeline(mode: str) -> StableDiffusionPipeline:
    """
    Поднимает генератор картинок и настраивает его под режим.

    Первый запуск скачивает веса в кеш hugging face: около 2 гигабайт на саму
    модель в половинной точности и 130 мегабайт на адаптер быстрого режима.

    Аргументы:
        mode: fast - четыре шага через lcm, quality - полный график dpm-solver.

    Возвращает:
        Готовый к вызову генератор на выбранном устройстве.
    """
    device, dtype = pick_device()

    # Фильтр запрещённого контента отключён: это ещё 1.2 гигабайта весов
    # ради проверки, которой на портретах нечего ловить.
    #
    # variant fp16 берёт из репозитория половинную копию весов: 2 гигабайта
    # загрузки вместо 4.3. На процессоре она разворачивается до float32 при
    # чтении, счёт от этого не меняется.
    pipeline = StableDiffusionPipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        torch_dtype = dtype,
        variant = "fp16",
        safety_checker = None,
        requires_safety_checker = False,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable = True)

    if mode == "fast":
        pipeline.load_lora_weights(LCM_LORA_ID)
        pipeline.fuse_lora()
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    print(f"--- генератор: {DIFFUSION_MODEL_ID} [{mode}] на {device}\n", flush = True)

    return pipeline


def count_prompt_tokens(pipeline: StableDiffusionPipeline, prompt: str) -> int:
    """
    Считает длину промпта в токенах текстового энкодера.

    Аргументы:
        pipeline: генератор картинок вместе со своим токенизатором.
        prompt: тег-промпт.

    Возвращает:
        Число токенов вместе со служебными началом и концом строки.
    """
    return len(pipeline.tokenizer(prompt, truncation = False).input_ids)


def draw_avatar(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    seed: int,
    steps: int,
    guidance: float,
) -> Image.Image:
    """
    Рисует одну аватарку по тег-промпту.

    Аргументы:
        pipeline: готовый генератор картинок.
        prompt: тег-промпт на английском.
        seed: зерно случайных чисел; одно зерно и один промпт дают одну картинку.
        steps: число шагов шумоподавления.
        guidance: сила следования промпту.

    Возвращает:
        Готовую картинку.
    """
    # Генератор случайных чисел держится на процессоре: на mps своя ветка
    # реализации, и то же зерно там даёт другую картинку.
    generator = torch.Generator(device = "cpu").manual_seed(seed)

    result = pipeline(
        prompt = prompt,
        num_inference_steps = steps,
        guidance_scale = guidance,
        width = IMAGE_SIDE,
        height = IMAGE_SIDE,
        generator = generator,
    )

    return result.images[0]


def draw_avatars(
    pipeline: StableDiffusionPipeline,
    prompts: dict[Path, str],
    output_directory: Path,
    seed: int,
    steps: int,
    guidance: float,
) -> None:
    """
    Рисует аватарку по каждому тег-промпту и складывает результат на диск.

    Рядом с картинкой пишется текстовый файл с промптом и зерном: без них
    повторить удачный кадр нечем.

    Аргументы:
        pipeline: готовый генератор картинок.
        prompts: тег-промпты по путям к исходным картинкам.
        output_directory: каталог для готовых аватарок.
        seed: зерно случайных чисел.
        steps: число шагов шумоподавления.
        guidance: сила следования промпту.

    Возвращает:
        Ничего.
    """
    output_directory.mkdir(parents = True, exist_ok = True)

    for image_path, prompt in prompts.items():
        token_count = count_prompt_tokens(pipeline = pipeline, prompt = prompt)
        if token_count > PROMPT_TOKEN_LIMIT:
            cut = token_count - PROMPT_TOKEN_LIMIT
            print(f"{image_path}: промпт {token_count} токенов, хвост в {cut} отброшен", flush = True)

        started = time.monotonic()
        picture = draw_avatar(
            pipeline = pipeline,
            prompt = prompt,
            seed = seed,
            steps = steps,
            guidance = guidance,
        )
        spent_seconds = time.monotonic() - started

        # Имена файлов в папках персонажей повторяются, поэтому в имя
        # результата входит и папка.
        stem = f"{image_path.parent.name}_{image_path.stem}_{seed}"
        picture_path = output_directory / f"{stem}.png"
        picture.save(picture_path)
        (output_directory / f"{stem}.txt").write_text(
            f"{prompt}\nseed={seed} steps={steps} guidance={guidance}\n",
            encoding = "utf-8",
        )

        print(f"{picture_path}: аватарка за {spent_seconds:.1f} с, токенов {token_count}\n", flush = True)


def main() -> None:
    """
    Прогоняет картинки по указанному пути через зрение, сборку промпта и генератор.

    Возвращает:
        Ничего.
    """
    setup_console_output()
    parser = argparse.ArgumentParser(description = "Проверка цепочки облик - промпт - аватарка")
    parser.add_argument("path", help = "файл с картинкой или каталог с картинками")
    parser.add_argument("--limit", type = int, default = 0, help = "сколько взять; ноль - все")
    parser.add_argument(
        "--mode",
        choices = ("fast", "quality"),
        default = "fast",
        help = "fast - четыре шага через lcm, quality - двадцать пять шагов",
    )
    parser.add_argument("--seed", type = int, default = 1, help = "зерно случайных чисел")
    parser.add_argument(
        "--out",
        default = str(PROJECT_ROOT / "docs" / "persona_avatars"),
        help = "каталог для готовых аватарок",
    )
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
    print(f"--- промпт: {writing_llm.model_name}, описаний {len(looks)}\n", flush = True)
    prompts = collect_prompts(llm = writing_llm, looks = looks)

    if not prompts:
        print("рисовать нечего", flush = True)
        return

    steps = FAST_STEPS if arguments.mode == "fast" else QUALITY_STEPS
    guidance = FAST_GUIDANCE if arguments.mode == "fast" else QUALITY_GUIDANCE

    pipeline = build_pipeline(mode = arguments.mode)
    draw_avatars(
        pipeline = pipeline,
        prompts = prompts,
        output_directory = Path(arguments.out),
        seed = arguments.seed,
        steps = steps,
        guidance = guidance,
    )


if __name__ == "__main__":
    main()
