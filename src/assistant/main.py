"""
Точка входа: вопрос из командной строки, ответ на экран.
"""

import argparse

from assistant.graph.graph import describe_nodes, run_research
from assistant.integrations.llm.client import build_provider_config
from assistant.variables import LLM_PROVIDER


def main() -> None:
    """
    Разбирает аргументы командной строки и печатает ответ ресёрчера.

    Возвращает:
        Ничего.
    """
    parser = argparse.ArgumentParser(description = "Ресёрчер: поиск в интернете и ответ по источникам")
    parser.add_argument("question", help = "вопрос, на который нужно ответить")
    arguments = parser.parse_args()

    print(f"[модель] {build_provider_config(provider = LLM_PROVIDER).model}")
    for line in describe_nodes():
        print(f"  {line}")

    answer = run_research(question = arguments.question)

    print("\n=== ответ ===")
    print(answer.answer)
    print(f"\nуверенность: {answer.confidence}")
    print("источники:")
    for url in answer.sources:
        print(f"  - {url}")


if __name__ == "__main__":
    main()
