"""
Замеры длительности этапов прогона.

Stopwatch копит пары «этап, секунды» через контекстный менеджер stage и
собирает из них таблицу. Замеры не печатаются по ходу: их забирает тот, кто
собрал прогон.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Stopwatch:
    """Копилка длительностей этапов прогона."""

    def __init__(self) -> None:
        self._stages: list[tuple[str, float]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """
        Замеряет длительность блока и кладёт её в копилку.

        Аргументы:
            name: имя этапа для таблицы.

        Возвращает:
            Контекстный менеджер без значения.
        """
        started = time.monotonic()
        try:
            yield
        finally:
            self._stages.append((name, time.monotonic() - started))

    def stages(self) -> list[tuple[str, float]]:
        """
        Отдаёт замеры в порядке прохождения.

        Возвращает:
            Пары «имя этапа, секунды».
        """
        return list(self._stages)

    def total_seconds(self) -> float:
        """
        Считает сумму замеров.

        Возвращает:
            Секунды всех этапов.
        """
        return sum(seconds for _name, seconds in self._stages)

    def render_table(self) -> str:
        """
        Собирает таблицу замеров.

        Возвращает:
            Строки «этап - секунды» и строку итога. Пустую строку, если замеров
            не было.
        """
        if not self._stages:
            return ""

        name_width = max(len(name) for name, _seconds in self._stages)
        lines = [f"{name:<{name_width}}  {seconds:6.1f} с" for name, seconds in self._stages]
        lines.append(f"{'итого':<{name_width}}  {self.total_seconds():6.1f} с")

        return "\n".join(lines)
