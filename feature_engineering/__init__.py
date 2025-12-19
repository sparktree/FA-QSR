"""
Feature Engineering Module

Provides feature extraction for spatial role labeling and
preposition disambiguation tasks.

Includes:
- Linguistic feature extraction (POS, dependencies, etc.)
- Spatial feature patterns
- Object affordance features
"""

from .spatial_features import (
    SpatialFeatureExtractor,
    FeatureVector,
    DependencyFeatures,
    LexicalFeatures,
    ContextFeatures,
)

__all__ = [
    'SpatialFeatureExtractor',
    'FeatureVector',
    'DependencyFeatures',
    'LexicalFeatures',
    'ContextFeatures',
]
