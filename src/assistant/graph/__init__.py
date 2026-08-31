"""
Граф ресёрчера: сбор фактов по источникам и изложение их в запрошенном виде.

Клиенты моделей приходят снаружи - см. contracts. Инструменты свои, в tools.
"""

from .contracts import NodeLlms, ResearchNode
from .graph import build_graph
from .state import Answer, ResearchNotes, ResearchState, Section
from .tools import CALL_BLOCKED, CALL_COMPLETED, CALL_FAILED, RESEARCH_TOOLS

__all__ = [
    "build_graph",
    "NodeLlms",
    "ResearchNode",
    "RESEARCH_TOOLS",
    "CALL_COMPLETED",
    "CALL_FAILED",
    "CALL_BLOCKED",
    "Answer",
    "ResearchNotes",
    "ResearchState",
    "Section",
]
