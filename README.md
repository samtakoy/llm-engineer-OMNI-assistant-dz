```bash
uv run python src/assistant/main.py

# или
uv run start-bot
```

Запуск
Модель в LM Studio должна быть загружена (qwen/qwen3.6-35b-a3b, сервер на localhost:1234).
```
# обычный прогон
.venv/bin/python -m assistant.main "Когда вышел Django 6.0?"

# с размышлением модели в логе
SHOW_REASONING=1 .venv/bin/python -m assistant.main "Когда вышел Django 6.0?"

# Через uv, если venv не активирован:
uv run python -m assistant.main "вопрос"

# Есть и entry point из pyproject.toml:
uv run start-bot "вопрос"
```


## Голос на входе

```
# вопрос из файла: формат любой, wav, m4a, aiff
.venv/bin/python -m assistant.main --audio recordings/2026-08-27T19-03-11.wav

# запись с микрофона до нажатия Enter
.venv/bin/python -m assistant.main --record

# запись фиксированной длительности
.venv/bin/python -m assistant.main --record 15
```

Через uv, если venv не активирован:

```
uv run python -m assistant.main --audio recordings/2026-08-27T19-03-11.wav
uv run python -m assistant.main --record
uv run python -m assistant.main --record 15

# то же через entry point из pyproject.toml
uv run start-bot --record 15
```

Запись сохраняется в `recordings/` именем по дате, путь печатается: тот же
вопрос дальше гоняется через `--audio` без микрофона.

Первый запуск качает модель распознавания в `~/.cache/huggingface`: `medium` -
около 1.5 ГБ. Дальше распознавание короткого вопроса занимает секунды, а повтор
по тому же файлу берётся из кеша расшифровок.

### Микрофон не слышно

Строка `[запись] ... громкость 0.00` и сообщение про тишину означают, что
терминалу не выдано разрешение на микрофон.

Системные настройки → Конфиденциальность и безопасность → Микрофон, включить то
приложение, из которого идёт запуск: Терминал, iTerm или Visual Studio Code.
Разрешение спрашивается у процесса-родителя, а не у python.

Если приложения в списке нет, диалог был отклонён раньше. Вернуть запрос:

```
tccutil reset Microphone com.microsoft.VSCode
```

Вместо `com.microsoft.VSCode` подставить свой: `com.apple.Terminal`,
`com.googlecode.iterm2`. После сброса приложение нужно перезапустить.

Проверить, какие устройства ввода видны:

```
.venv/bin/python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Другие системы

Библиотеки распознавания ставятся готовыми колёсами на macOS, Linux и Windows.
Платформенная зависимость одна - запись с микрофона.

На linux колесо `sounddevice` идёт без portaudio и ищет системную библиотеку:

```
sudo apt install libportaudio2
```

Без неё `--record` вернёт «библиотека записи недоступна», а `--audio` и весь
остальной прогон работают как обычно.

На linux с видеокартой nvidia `SPEECH_DEVICE=auto` выберет cuda, и `ctranslate2`
потребует cuDNN 9 и cuBLAS. Нет их - модель не загрузится с внятной причиной,
лечится `SPEECH_DEVICE=cpu` или пакетом `nvidia-cudnn-cu12`. Там же осмысленнее
`SPEECH_COMPUTE_TYPE=float16`: `int8` выбран под процессор.

На windows ставится как есть, ничего доустанавливать не нужно.


## Переменные окружения

Читаются один раз в `src/assistant/variables.py` из `.env`. Все со значениями по
умолчанию, работать можно без `.env` вовсе.

### Модель

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `LLM_PROVIDER` | `local` | Провайдер: `local`, `openrouter`, `openai`, `yc`. |
| `LOCAL_BASE_URL` | `http://localhost:1234/v1` | Адрес сервера lm studio. |
| `LOCAL_API_KEY` | `lm-studio` | Ключ локального сервера, любой непустой. |
| `LOCAL_MODEL` | `google/gemma-4-26b-a4b-qat` | Имя модели в lm studio. |
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
| `SHOW_REASONING` | выкл | Показывать размышление модели в фазе поиска. Дороже примерно вчетверо по времени. |

### Веб-слой и его кеш

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `WEB_USER_AGENT` | `omni-assistant/0.1 (+ссылка)` | Строка User-Agent. Wikimedia и часть площадок отвечают 403 на agent без контакта владельца бота. |
| `WEB_CACHE_DIR` | `.cache/web` | Каталог кеша страниц и выдачи. Пусто выключает кеш. Относительный путь считается от корня проекта. |
| `WEB_PAGE_CACHE_TTL_DAYS` | `0` | Срок годности текста страницы в днях. Ноль - не протухает. |
| `WEB_SEARCH_CACHE_TTL_DAYS` | `0` | Срок годности поисковой выдачи в днях. Ноль - не протухает. |
| `WEB_CACHE_BYPASS` | выкл | Читать мимо кеша. Запись продолжается, поэтому прогон обновляет хранилище. |

### Речевой слой и его кеш

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
WEB_CACHE_BYPASS=1 .venv/bin/python -m assistant.main "вопрос"
```

### Речевой выход

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `SPEAKING_MODEL_ID` | `v5_5_ru` | Версия модели silero. Набор голосов принадлежит версии. |
| `SPEAKING_LANGUAGE` | `ru` | Язык модели синтеза. |
| `SPEAKING_DEVICE` | `auto` | Устройство синтеза: `auto`, `cpu`, `cuda`. |
| `SPEAKING_SAMPLE_RATE` | `48000` | Частота дискретизации: `8000`, `24000`, `48000`. |
| `SPEAKING_PUT_ACCENT` | вкл | Расставлять ударения. Действует только при озвучке чистым текстом. |
| `SPEAKING_PUT_YO` | вкл | Восстанавливать букву ё. Ограничение то же. |
| `SPEAKING_HUB_DIR` | `.cache/silero` | Куда torch.hub кладёт код silero и файл весов. Пусто - каталог torch по умолчанию. |
| `SPOKEN_DIR` | `spoken` | Куда складывать озвученные файлы. |

## Рассказчик

```
.venv/bin/python -m assistant.main "вопрос"

.venv/bin/python -m assistant.main "ты гид по пятигорску. нужно  собрать информацию для экскурсии на гору машук в пятигорске и составить красочное описание по маршруту для детей 7-8 лет" --narrator "угрюмый злой халк"

.venv/bin/python -m assistant.main "ты гид по пятигорску. нужно  собрать информацию для экскурсии на гору машук в пятигорске и составить красочное описание по маршруту для детей 7-8 лет" --image docs/persona_images/hulk/images.jpeg

.venv/bin/python -m assistant.main "ты гид по пятигорску. нужно  собрать информацию для экскурсии на гору машук в пятигорске и составить красочное описание по маршруту для детей 7-8 лет" --image docs/persona_images/hulk/images.jpeg --persona-mode structured
```





## Голос на выходе

```
# экскурсия голосом персонажа с фотографии; текст перед озвучкой размечается
# паузами и ударениями
.venv/bin/python -m assistant.main "экскурсия по горе машук для детей 7-8 лет" --image docs/persona_images/hulk/images.jpeg --persona-mode structured --speak

# то же, но без разметки: озвучка идёт чистым текстом
.venv/bin/python -m assistant.main "экскурсия по горе машук для детей 7-8 лет" --image docs/persona_images/hulk/images.jpeg --persona-mode structured --speak --no-markup

# без фотографии: первый голос модели, нейтральные темп и высота
.venv/bin/python -m assistant.main "Когда вышел Django 6.0?" --speak
```

Озвучка идёт кусками - вступление, каждый раздел, завершение. Путь к файлу
печатается сразу, как кусок готов: первый можно слушать, пока считаются
остальные. Файлы падают в `spoken/` именем по дате и номеру куска.

Первый запуск качает модель синтеза в `.cache/silero`: код silero и файл весов,
для `v5_5_ru` около 145 МБ. Дальше синтез идёт быстрее реального времени, а
время съедает разметка: она стоит вызова модели на каждый кусок.

## Прогон с чекпоинта

```
# какие прогоны можно переиграть
.venv/bin/python -m assistant.main --list-runs

# переиграть изложение по собранным фактам
.venv/bin/python -m assistant.main --resume 20260829-160711 --from compose

# то же, но другим рассказчиком
.venv/bin/python -m assistant.main --resume 20260829-160711 --from compose --narrator "добрый краевед"

# пересобрать факты и изложение
.venv/bin/python -m assistant.main --resume 20260829-160711 --from collect
```
