"""
Network Translator Module

Translates linguistic spatial representations (SpRL output, GUM annotations)
into FA-QSR constraint networks for formal reasoning.

This module implements the bridge between:
- NLP pipeline outputs (spatial triples)
- GUM ontology spatial modalities
- FA-QSR constraint networks

The translation preserves both geometric and functional semantics.
"""

from .gum_translator import (
    GUMToFAQSRTranslator,
    TranslationResult,
    TranslationConfig,
)
from .sprl_to_network import (
    SpRLToNetworkTranslator,
    NetworkBuilder,
)

__all__ = [
    'GUMToFAQSRTranslator',
    'TranslationResult',
    'TranslationConfig',
    'SpRLToNetworkTranslator',
    'NetworkBuilder',
]
