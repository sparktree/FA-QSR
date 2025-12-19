"""
NLP Models Module

Machine learning models for spatial language understanding:
- Spatial Role Labeling (SpRL)
- Preposition Sense Disambiguation

Models are designed to be trained on SpaceEval/SpRL annotated corpora
and output structured spatial relations for FA-QSR processing.
"""

from .sprl_model import (
    SpatialRoleLabeler,
    SpRLAnnotation,
    SpatialTriple,
)
from .preposition_model import (
    PrepositionDisambiguator,
    PrepositionSense,
)

__all__ = [
    'SpatialRoleLabeler',
    'SpRLAnnotation',
    'SpatialTriple',
    'PrepositionDisambiguator',
    'PrepositionSense',
]
