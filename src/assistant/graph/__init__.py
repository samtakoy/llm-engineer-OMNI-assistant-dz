"""
Граф ресёрчера: сбор фактов по источникам и изложение их в запрошенном виде.
"""

from .graph import build_graph, run_research
from .llms import describe_nodes
from .state import Answer, ResearchNotes, ResearchState, Section

__all__ = [
    "build_graph",
    "run_research",
    "describe_nodes",
    "Answer",
    "ResearchNotes",
    "ResearchState",
    "Section",
]
