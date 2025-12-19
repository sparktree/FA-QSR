"""
FA-QSR Configuration Module

Centralized configuration for the FA-QSR framework.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass
class PathConfig:
    """File path configuration."""
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = field(default="")
    ontology_dir: str = field(default="")
    results_dir: str = field(default="")
    models_dir: str = field(default="")

    def __post_init__(self):
        self.data_dir = os.path.join(self.project_root, "data")
        self.ontology_dir = os.path.join(self.project_root, "ontology")
        self.results_dir = os.path.join(self.project_root, "results")
        self.models_dir = os.path.join(self.project_root, "nlp_models")


@dataclass
class NLPConfig:
    """NLP pipeline configuration."""
    # SpRL settings
    use_heuristic_labeling: bool = True
    min_confidence_threshold: float = 0.5

    # Preposition disambiguation
    use_affordance_features: bool = True
    use_context_features: bool = True

    # Feature extraction
    context_window_size: int = 3
    use_dependency_features: bool = True


@dataclass
class ReasoningConfig:
    """Reasoning engine configuration."""
    # Path consistency
    max_iterations: int = 10000
    enable_early_termination: bool = True

    # Backtracking
    enable_backtracking: bool = False
    max_backtrack_nodes: int = 100000
    use_variable_ordering: str = "mrv"  # mrv, degree, first
    use_value_ordering: str = "first"   # first, lcv


@dataclass
class ComplexityConfig:
    """Complexity analysis configuration."""
    analyze_by_default: bool = True
    warn_on_np_hard: bool = True
    compute_tractable_fraction: bool = True


@dataclass
class FAQSRGlobalConfig:
    """Global FA-QSR configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'paths': {
                'project_root': self.paths.project_root,
                'data_dir': self.paths.data_dir,
                'ontology_dir': self.paths.ontology_dir,
            },
            'nlp': {
                'use_heuristic_labeling': self.nlp.use_heuristic_labeling,
                'min_confidence_threshold': self.nlp.min_confidence_threshold,
            },
            'reasoning': {
                'max_iterations': self.reasoning.max_iterations,
                'enable_backtracking': self.reasoning.enable_backtracking,
            },
            'complexity': {
                'analyze_by_default': self.complexity.analyze_by_default,
                'warn_on_np_hard': self.complexity.warn_on_np_hard,
            },
            'log_level': self.log_level,
            'verbose': self.verbose,
        }

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'FAQSRGlobalConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = cls()
        # Would populate from data dict
        return config


# Default global configuration instance
DEFAULT_CONFIG = FAQSRGlobalConfig()


# GUM Ontology URIs
GUM_NAMESPACE = "http://www.ontospace.uni-bremen.de/ontology/stable/GUM-3.owl#"
GUM_SPACE_NAMESPACE = "http://www.ontospace.uni-bremen.de/ontology/stable/GUM-3-space.owl#"
FAQSR_NAMESPACE = "http://faqsr.spatial-reasoning.org/ontology/faqsr#"


# Spatial preposition lexicon
SPATIAL_PREPOSITIONS = {
    # Topological/Functional
    'in', 'inside', 'within', 'into',
    'on', 'onto', 'upon', 'atop',
    'at', 'by', 'beside', 'alongside',
    'near', 'close to', 'next to',
    'under', 'beneath', 'below', 'underneath',
    'over', 'above',
    'behind', 'after', 'in back of',
    'before', 'ahead of', 'in front of',
    'between', 'among', 'amongst',
    'through', 'across', 'along',
    'around', 'about',
    'against', 'toward', 'towards',
    'from', 'off', 'out of', 'away from',
    'outside',
}

# Motion verbs relevant to spatial language
MOTION_VERBS = {
    'go', 'come', 'move', 'travel', 'walk', 'run',
    'enter', 'exit', 'leave', 'arrive', 'depart',
    'put', 'place', 'set', 'lay', 'position',
    'push', 'pull', 'slide', 'roll',
    'rise', 'fall', 'climb', 'descend',
    'cross', 'pass', 'reach', 'approach',
    'hang', 'suspend', 'attach', 'mount',
}

# Object affordance knowledge base
OBJECT_AFFORDANCES = {
    # Containers
    'box': {'containment': True, 'support': False, 'type': 'container'},
    'bag': {'containment': True, 'support': False, 'type': 'container'},
    'cup': {'containment': True, 'support': False, 'type': 'container'},
    'bowl': {'containment': True, 'support': False, 'type': 'container'},
    'vase': {'containment': True, 'support': True, 'type': 'container'},
    'basket': {'containment': True, 'support': False, 'type': 'container'},
    'drawer': {'containment': True, 'support': True, 'type': 'container'},
    'bottle': {'containment': True, 'support': False, 'type': 'container'},
    'jar': {'containment': True, 'support': False, 'type': 'container'},

    # Horizontal support surfaces
    'table': {'containment': False, 'support': True, 'orientation': 'horizontal'},
    'desk': {'containment': False, 'support': True, 'orientation': 'horizontal'},
    'shelf': {'containment': False, 'support': True, 'orientation': 'horizontal'},
    'floor': {'containment': False, 'support': True, 'orientation': 'horizontal'},
    'ground': {'containment': False, 'support': True, 'orientation': 'horizontal'},
    'counter': {'containment': False, 'support': True, 'orientation': 'horizontal'},
    'bench': {'containment': False, 'support': True, 'orientation': 'horizontal'},

    # Vertical surfaces
    'wall': {'containment': False, 'support': True, 'orientation': 'vertical'},
    'door': {'containment': False, 'support': True, 'orientation': 'vertical'},
    'board': {'containment': False, 'support': True, 'orientation': 'vertical'},
    'ceiling': {'containment': False, 'support': True, 'orientation': 'vertical'},

    # Hanging supports
    'hook': {'containment': False, 'support': True, 'type': 'hanging'},
    'peg': {'containment': False, 'support': True, 'type': 'hanging'},
    'hanger': {'containment': False, 'support': True, 'type': 'hanging'},
    'nail': {'containment': False, 'support': True, 'type': 'hanging'},
    'branch': {'containment': False, 'support': True, 'type': 'hanging'},

    # Regions
    'room': {'containment': True, 'support': False, 'type': 'region'},
    'house': {'containment': True, 'support': False, 'type': 'region'},
    'building': {'containment': True, 'support': False, 'type': 'region'},
    'city': {'containment': True, 'support': False, 'type': 'region'},
    'country': {'containment': True, 'support': False, 'type': 'region'},
    'garden': {'containment': True, 'support': False, 'type': 'region'},
    'park': {'containment': True, 'support': False, 'type': 'region'},
    'kitchen': {'containment': True, 'support': False, 'type': 'region'},
}
