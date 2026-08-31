# Переменные окружения

Читаются один раз в `src/assistant/variables.py` из `.env`. Все со значениями по
умолчанию, работать можно без `.env`.

## Быстрая минимальная настройка

```
# провайдер текстовых моделей
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=qwen3.6:35b-a3b

# модель зрения: разбирает фотографию персонажа
VISION_PROVIDER=ollama
VISION_MODEL=qwen3-vl:4b

# контакт владельца бота: без него часть площадок отвечает 403
WEB_USER_AGENT=omni-assistant/0.1 (+https://github.com/samtakoy)
```

Остальное берётся по умолчанию: кеши, речевой вход и выход, каталоги вывода.

Для lm studio вместо блока ollama:

```
LLM_PROVIDER=local
LOCAL_BASE_URL=http://localhost:1234/v1
LOCAL_MODEL=qwen/qwen3.6-35b-a3b
VISION_PROVIDER=local
VISION_MODEL=qwen/qwen3-vl-4b
```

## Провайдер моделей

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `LLM_PROVIDER` | `local` | Провайдер: `local`, `ollama`, `openrouter`, `openai`, `yc`. |
| `LOCAL_BASE_URL` | `http://localhost:1234/v1` | Адрес сервера lm studio. |
| `LOCAL_API_KEY` | `lm-studio` | Ключ локального сервера, любой непустой. |
| `LOCAL_MODEL` | `google/gemma-4-26b-a4b-qat` | Имя модели в lm studio. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Адрес сервера ollama. |
| `OLLAMA_API_KEY` | `ollama` | Ключ сервера ollama, любой непустой. |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Имя модели в ollama вместе с тегом. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Адрес openrouter. |
| `OPENROUTER_API_KEY` | пусто | Ключ openrouter. |
| `OPENROUTER_MODEL` | `z-ai/glm-4.7-flash` | Модель openrouter. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Адрес openai. |
| `OPENAI_API_KEY` | пусто | Ключ openai. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Модель openai. |
| `YC_BASE_URL` | `https://llm.api.cloud.yandex.net/v1` | Адрес yandex cloud. |
| `YC_API_KEY` | пусто | Ключ yandex cloud. |
| `YC_FOLDER_ID` | пусто | Каталог yandex cloud. |
| `YC_MODEL` | `yandexgpt-lite` | Модель yandex cloud. |
| `LLM_TEMPERATURE` | пусто | Перекрывает температуру профиля модели. |
| `LLM_SEED` | пусто | Фиксирует зерно генерации ради повторяемости. |
| `ENABLE_ALL_REASONING` | выкл | Включает размышление на всех узлах. Прогон заметно дольше, режим отладочный. |

## Модель зрения

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `VISION_PROVIDER` | значение `LLM_PROVIDER` | Провайдер модели, разбирающей фотографию. |
| `VISION_MODEL` | `qwen/qwen3-vl-4b` | Имя мультимодальной модели. |
| `VISION_MAX_SIDE` | `1024` | До какого размера ужимать картинку перед отправкой. |
| `VISION_JPEG_QUALITY` | `85` | Качество пережатия картинки. |
| `VISION_CACHE_DIR` | `.cache/vision` | Каталог кеша разборов картинок. Пусто выключает кеш. |
| `VISION_CACHE_TTL_DAYS` | `0` | Срок годности разбора в днях. Ноль - не протухает. |
| `VISION_CACHE_BYPASS` | выкл | Разбирать картинку заново, минуя кеш. |

## Рассказчик

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `PERSONA_MODE` | `structured` | Как строить рассказчика по фотографии: `free` - одной фразой, `structured` - полями схемы. |

## Веб-слой и его кеш

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `WEB_USER_AGENT` | `omni-assistant/0.1 (+ссылка)` | Строка User-Agent. Wikimedia и часть площадок отвечают 403 на agent без контакта владельца бота. |
| `WEB_CACHE_DIR` | `.cache/web` | Каталог кеша страниц и выдачи. Пусто выключает кеш. Относительный путь считается от корня проекта. |
| `WEB_PAGE_CACHE_TTL_DAYS` | `0` | Срок годности текста страницы в днях. Ноль - не протухает. |
| `WEB_SEARCH_CACHE_TTL_DAYS` | `0` | Срок годности поисковой выдачи в днях. Ноль - не протухает. |
| `WEB_CACHE_BYPASS` | выкл | Читать мимо кеша. Запись продолжается, поэтому прогон обновляет хранилище. |

## Журнал и снимки прогона

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `TRACE_DIR` | `logs/traces` | Каталог журналов прогона. Пусто выключает журнал. |
| `CHECKPOINT_DIR` | `.cache/checkpoints` | Каталог снимков состояния графа. Пусто выключает снимки, и переиграть прогон нельзя. |

## Речевой вход и его кеш

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `SPEECH_MODEL` | `medium` | Модель faster-whisper: `small`, `medium`, `large-v3`. На русском `small` путает имена собственные. |
| `SPEECH_DEVICE` | `auto` | Устройство вычислений: `auto`, `cpu`, `cuda`. |
| `SPEECH_COMPUTE_TYPE` | `int8` | Тип вычислений: `int8` на процессоре, `float16` на видеокарте. |
| `SPEECH_LANGUAGE` | `ru` | Язык записи. Пусто - определять по звуку. |
| `SPEECH_CACHE_DIR` | `.cache/speech` | Каталог кеша расшифровок. Пусто выключает кеш. |
| `SPEECH_CACHE_TTL_DAYS` | `0` | Срок годности расшифровки в днях. Ноль - не протухает. |
| `SPEECH_CACHE_BYPASS` | выкл | Распознавать заново, минуя кеш. |
| `RECORDINGS_DIR` | `recordings` | Куда складывать записи с микрофона. |

Сроки годности нулевые не по недосмотру: кеш собирается ради воспроизводимых
прогонов, а короткий срок молча выбрасывал бы собранный материал. Чистка -
удалением каталога `.cache/`.

Переменные обхода кеша держать в `.env` незачем, это флаг на один запуск:

```
WEB_CACHE_BYPASS=1 uv run start-bot "вопрос"
```

## Речевой выход

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `SPEAKING_MODEL_ID` | `v5_5_ru` | Версия модели silero. Набор голосов принадлежит версии. |
| `SPEAKING_LANGUAGE` | `ru` | Язык модели синтеза. |
| `SPEAKING_DEVICE` | `auto` | Устройство синтеза: `auto`, `cpu`, `cuda`. |
| `SPEAKING_SAMPLE_RATE` | `48000` | Частота дискретизации: `8000`, `24000`, `48000`. |
| `SPEAKING_PUT_ACCENT` | вкл | Расставлять ударения. Действует только при озвучке чистым текстом. |
| `SPEAKING_PUT_YO` | вкл | Восстанавливать букву ё. Ограничение то же. |
| `SPEAKING_HUB_DIR` | `.cache/silero` | Куда torch.hub кладёт код silero и файл весов. Пусто - каталог torch по умолчанию. |
| `SPEAKING_MAX_SYMBOLS` | `900` | Бюджет символов без разметки на один кусок синтеза. Текст длиннее режется по предложениям. |
| `MALE_SPEAKER` | `eugene` | Мужской голос версии модели: запасной голос рассказчика и голос диктора. |
| `FEMALE_SPEAKER` | `xenia` | Женский голос той же роли. |
| `SPOKEN_DIR` | `spoken` | Куда складывать тексты прогона и озвученные файлы. |
