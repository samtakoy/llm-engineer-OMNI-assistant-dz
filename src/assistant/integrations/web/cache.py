"""
Хранилище ответов сети на диске.

Каждый прогон начинал сбор заново и выкачивал то же самое: выдача поисковика по
одному и тому же вопросу устойчива, и половина адресов повторялась из прогона в
прогон. Здесь ответ ложится на диск один раз и переиспользуется, пока не
устареет.

Файл на запись, а не общий индекс: прогон может оборваться на любой записи, и
одна страница не должна рисковать остальными. Имя файла - отпечаток ключа, сам
ключ лежит внутри записи, поэтому по каталогу видно, что в нём хранится, и
совпадение отпечатков ловится при чтении.

Неудачи не запоминаются вызывающим кодом: недоступный сайт через неделю может
отвечать, а запись об отказе стоила бы живого ответа.

Ни один метод не выбрасывает исключений наружу: битая запись, нечитаемый
каталог и полный диск дают промах и строку на экран, а не падение прогона.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class FileCache:
    """
    Каталог записей вида «ключ → полезная нагрузка».

    Формат записи: версия, ключ, дата сохранения и поля нагрузки на одном
    уровне. Версия растёт, когда меняется состав полей: старые записи тогда
    перестают читаться и данные добываются заново, вместо того чтобы приехать
    в код половиной ожидаемых полей.
    """

    def __init__(self, directory: Path, ttl_days: int, record_version: int) -> None:
        """
        Аргументы:
            directory: каталог, в котором лежат записи.
            ttl_days: сколько дней запись считается годной. Ноль и меньше -
                годна всегда.
            record_version: версия формата записи.
        """
        self._directory = directory
        self._ttl_days = ttl_days
        self._record_version = record_version

    def read(self, key: str) -> dict[str, Any] | None:
        """
        Читает годную запись по ключу.

        Аргументы:
            key: ключ записи.

        Возвращает:
            Запись целиком либо None, если записи нет, она битая, сделана
            несовместимой версией формата, принадлежит другому ключу или
            устарела.
        """
        path = self._entry_path(key = key)
        if not path.exists():
            return None

        try:
            record = json.loads(path.read_text(encoding = "utf-8"))
        except Exception as error:
            # Битую запись лечит повторное обращение к сети, а не разбор причины.
            print(f"[cache] запись не прочитана {path.name}: {type(error).__name__}: {error}")
            return None

        if not isinstance(record, dict) or record.get("version") != self._record_version:
            return None
        if record.get("key") != key:
            # Совпадение отпечатков: ключ в записи чужой, нагрузка к запрошенному не относится.
            return None
        if not self._is_fresh(record = record):
            return None

        return record

    def write(self, key: str, payload: dict[str, Any]) -> None:
        """
        Кладёт нагрузку в хранилище, затирая прежнюю запись по этому ключу.

        Аргументы:
            key: ключ записи.
            payload: поля нагрузки; попадают в запись рядом со служебными.

        Возвращает:
            Ничего. Неудача записи печатается и прогон не роняет.
        """
        record = {
            "version": self._record_version,
            "key": key,
            "stored_at": datetime.now().isoformat(timespec = "seconds"),
            **payload,
        }

        path = self._entry_path(key = key)
        try:
            self._directory.mkdir(parents = True, exist_ok = True)
            # Запись через временный файл: прогон обрывается в любой момент, и
            # половина json на диске читалась бы как битая запись до самой чистки.
            temporary = path.with_suffix(".part")
            temporary.write_text(json.dumps(record, ensure_ascii = False), encoding = "utf-8")
            temporary.replace(path)
        except Exception as error:
            print(f"[cache] запись не сохранена {key}: {type(error).__name__}: {error}")

    def _entry_path(self, key: str) -> Path:
        """
        Составляет путь к файлу записи.

        Аргументы:
            key: ключ записи.

        Возвращает:
            Путь к файлу.
        """
        fingerprint = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self._directory / f"{fingerprint}.json"

    def _is_fresh(self, record: dict[str, Any]) -> bool:
        """
        Проверяет, не устарела ли запись.

        Аргументы:
            record: прочитанная запись.

        Возвращает:
            True, если запись ещё можно использовать.
        """
        if self._ttl_days <= 0:
            return True

        try:
            stored_at = datetime.fromisoformat(record["stored_at"])
        except Exception:
            return False

        return datetime.now() - stored_at < timedelta(days = self._ttl_days)


def open_cache(
    directory: Path | None,
    namespace: str,
    ttl_days: int,
    record_version: int,
) -> FileCache | None:
    """
    Открывает хранилище, если кеш включён.

    Аргументы:
        directory: корневой каталог кеша; None означает выключенный кеш.
        namespace: подкаталог под свой вид записей. Разные виды не смешиваются:
            у каждого свой формат и своя версия.
        ttl_days: сколько дней запись считается годной.
        record_version: версия формата записи.

    Возвращает:
        Хранилище либо None, если кеш выключен.
    """
    if directory is None:
        return None

    return FileCache(
        directory = directory / namespace,
        ttl_days = ttl_days,
        record_version = record_version,
    )
