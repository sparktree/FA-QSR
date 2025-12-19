"""
QSR Base Module - Qualitative Spatial Reasoning Foundation

This module provides the core qualitative spatial reasoning calculi
that form the geometric foundation of FA-QSR.

Includes:
- RCC-8: Region Connection Calculus (8 base relations)
- Composition tables for constraint propagation
- Tractable fragment analysis
"""

from .rcc8 import (
    RCC8Relation,
    RelationSet,
    relation_set,
    UNIVERSAL,
    EMPTY,
    BASIC,
    RCC8CompositionTable,
    COMPOSITION_TABLE,
    Constraint,
    ConstraintNetwork,
    TractableFragments,
    semantic_relations,
)

__all__ = [
    'RCC8Relation',
    'RelationSet',
    'relation_set',
    'UNIVERSAL',
    'EMPTY',
    'BASIC',
    'RCC8CompositionTable',
    'COMPOSITION_TABLE',
    'Constraint',
    'ConstraintNetwork',
    'TractableFragments',
    'semantic_relations',
]
