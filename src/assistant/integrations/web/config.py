"""
Настройки веб-слоя.

Пакет не читает окружение и ничего не знает о проекте, в который встроен: все
значения приходят снаружи одним объектом. Так папку можно скопировать в другой
проект и собрать конфиг там, где этому проекту удобно.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class WebConfig:
    """
    Настройки поиска, загрузки страниц и кеша.

    Атрибуты:
        user_agent: строка User-Agent для запросов к сайтам. Часть площадок
            отвечает 403 на agent без контакта владельца бота.
        request_timeout_seconds: сколько ждать ответа сайта.
        max_page_bytes: сколько байт страницы скачивать, дальше поток обрывается.
        cache_directory: каталог кеша; None выключает кеш целиком.
        page_cache_ttl_days: сколько дней годна запись о странице. Ноль и меньше -
            годна всегда.
        search_cache_ttl_days: сколько дней годна запись о поисковой выдаче.
            Ноль и меньше - годна всегда.
        bypass_cache: читать мимо кеша. Запись при этом продолжается, поэтому
            прогон с обходом обновляет хранилище свежими данными.
    """

    user_agent: str
    request_timeout_seconds: float
    max_page_bytes: int
    cache_directory: Path | None
    page_cache_ttl_days: int
    search_cache_ttl_days: int
    bypass_cache: bool
