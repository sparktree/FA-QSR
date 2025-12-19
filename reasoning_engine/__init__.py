"""
Reasoning Engine Module

This module provides constraint propagation and consistency checking
algorithms for FA-QSR networks.

Includes:
- Path consistency (algebraic closure) algorithms
- Backtracking search for satisfiability
- Hybrid geometric-functional reasoning
"""

from .path_consistency import (
    PathConsistencyChecker,
    FAQSRReasoner,
    ConsistencyResult,
)
from .backtrack import BacktrackingSolver

__all__ = [
    'PathConsistencyChecker',
    'FAQSRReasoner',
    'ConsistencyResult',
    'BacktrackingSolver',
]
