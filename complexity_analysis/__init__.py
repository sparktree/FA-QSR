"""
Complexity Analysis Module

Analyzes computational complexity of FA-QSR constraint networks
to validate the cognitive-computational split between geometric
and functional reasoning.

Implements Phase 3 of the FA-QSR framework:
- Tractability analysis for pure geometric fragments
- NP-hardness identification for functional fragments
- Complexity gradient prediction
"""

from .complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityResult,
    FragmentClassifier,
    TractabilityChecker,
)

__all__ = [
    'ComplexityAnalyzer',
    'ComplexityResult',
    'FragmentClassifier',
    'TractabilityChecker',
]
