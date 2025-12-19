"""
Results and Evaluation Module

Provides evaluation metrics and result collection for FA-QSR experiments:
- NLP pipeline evaluation (SpRL precision/recall/F1)
- Reasoning engine performance metrics
- Complexity analysis validation
"""

from .evaluation import (
    Evaluator,
    EvaluationResult,
    MetricsCollector,
    ExperimentRunner,
)

__all__ = [
    'Evaluator',
    'EvaluationResult',
    'MetricsCollector',
    'ExperimentRunner',
]
