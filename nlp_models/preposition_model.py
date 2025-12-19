"""
Preposition Sense Disambiguation Model

This module disambiguates spatial preposition senses based on
trajector-landmark context. Different senses map to different
FA-QSR constraint patterns.

Key insight from Herskovits (1986) and Coventry et al. (2001):
The same preposition (e.g., "on") can express different spatial
relations depending on functional context:
- "The cup is on the table" - horizontal support
- "The picture is on the wall" - vertical attachment
- "The coat is on the hook" - hanging support

References:
- Herskovits (1986) - Language and Spatial Cognition
- Coventry et al. (2001) - Functional influences on spatial language
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.spatial_features import (
    Token,
    PrepositionSense,
    SpatialFeatureExtractor,
)
from nlp_models.sprl_model import SpatialTriple


@dataclass
class SenseDisambiguation:
    """
    Result of preposition sense disambiguation.

    Includes the predicted sense and confidence, plus alternative
    senses with their probabilities.
    """
    preposition: str
    predicted_sense: PrepositionSense
    confidence: float
    alternatives: Dict[PrepositionSense, float]

    # Context used for disambiguation
    trajector_lemma: str
    landmark_lemma: str

    # Functional indicators
    is_functional: bool  # Whether functional (vs purely geometric)
    support_type: Optional[str] = None
    containment_type: Optional[str] = None


class PrepositionDisambiguator:
    """
    Disambiguates spatial preposition senses.

    Uses trajector and landmark properties (especially affordances)
    to determine the appropriate semantic sense.
    """

    # Preposition -> possible senses mapping
    PREPOSITION_SENSES = {
        'on': [
            PrepositionSense.ON_SUPPORT,
            PrepositionSense.ON_VERTICAL,
            PrepositionSense.ON_HANGING,
        ],
        'in': [
            PrepositionSense.IN_CONTAINER,
            PrepositionSense.IN_REGION,
            PrepositionSense.IN_FUNCTIONAL,
        ],
        'at': [
            PrepositionSense.AT_LOCATION,
        ],
        'near': [
            PrepositionSense.NEAR_PROXIMITY,
        ],
        'by': [
            PrepositionSense.BY_ADJACENCY,
            PrepositionSense.NEAR_PROXIMITY,
        ],
        'above': [
            PrepositionSense.ABOVE_VERTICAL,
        ],
        'below': [
            PrepositionSense.BELOW_VERTICAL,
        ],
        'under': [
            PrepositionSense.BELOW_VERTICAL,
        ],
        'over': [
            PrepositionSense.ABOVE_VERTICAL,
        ],
        'behind': [
            PrepositionSense.BEHIND_PROJECTIVE,
        ],
        'in front of': [
            PrepositionSense.FRONT_PROJECTIVE,
        ],
    }

    # Landmark types that trigger specific senses
    LANDMARK_SENSE_TRIGGERS = {
        # Vertical surfaces -> vertical support sense for "on"
        'wall': {'on': PrepositionSense.ON_VERTICAL},
        'door': {'on': PrepositionSense.ON_VERTICAL},
        'ceiling': {'on': PrepositionSense.ON_VERTICAL},
        'board': {'on': PrepositionSense.ON_VERTICAL},

        # Hanging supports -> hanging sense for "on"
        'hook': {'on': PrepositionSense.ON_HANGING},
        'peg': {'on': PrepositionSense.ON_HANGING},
        'hanger': {'on': PrepositionSense.ON_HANGING},
        'nail': {'on': PrepositionSense.ON_HANGING},
        'branch': {'on': PrepositionSense.ON_HANGING},

        # Horizontal surfaces -> standard support for "on"
        'table': {'on': PrepositionSense.ON_SUPPORT},
        'desk': {'on': PrepositionSense.ON_SUPPORT},
        'shelf': {'on': PrepositionSense.ON_SUPPORT},
        'floor': {'on': PrepositionSense.ON_SUPPORT},
        'ground': {'on': PrepositionSense.ON_SUPPORT},

        # Containers -> container sense for "in"
        'box': {'in': PrepositionSense.IN_CONTAINER},
        'bag': {'in': PrepositionSense.IN_CONTAINER},
        'cup': {'in': PrepositionSense.IN_CONTAINER},
        'bowl': {'in': PrepositionSense.IN_CONTAINER},
        'vase': {'in': PrepositionSense.IN_CONTAINER},
        'basket': {'in': PrepositionSense.IN_CONTAINER},
        'drawer': {'in': PrepositionSense.IN_CONTAINER},
        'pocket': {'in': PrepositionSense.IN_CONTAINER},

        # Regions -> region sense for "in"
        'room': {'in': PrepositionSense.IN_REGION},
        'house': {'in': PrepositionSense.IN_REGION},
        'building': {'in': PrepositionSense.IN_REGION},
        'city': {'in': PrepositionSense.IN_REGION},
        'country': {'in': PrepositionSense.IN_REGION},
        'area': {'in': PrepositionSense.IN_REGION},
        'garden': {'in': PrepositionSense.IN_REGION},
    }

    # Functional sense indicators based on landmark affordances
    AFFORDANCE_INDICATORS = {
        ('support', 'horizontal'): PrepositionSense.ON_SUPPORT,
        ('support', 'vertical'): PrepositionSense.ON_VERTICAL,
        ('support', 'hanging'): PrepositionSense.ON_HANGING,
        ('containment', 'physical'): PrepositionSense.IN_CONTAINER,
        ('containment', 'regional'): PrepositionSense.IN_REGION,
        ('containment', 'functional'): PrepositionSense.IN_FUNCTIONAL,
    }

    def __init__(self):
        self.feature_extractor = SpatialFeatureExtractor()

    def disambiguate(self, triple: SpatialTriple,
                    tokens: Optional[List[Token]] = None) -> SenseDisambiguation:
        """
        Disambiguate the preposition sense for a spatial triple.

        Args:
            triple: The spatial triple (trajector, indicator, landmark)
            tokens: Optional token list for additional context

        Returns:
            SenseDisambiguation with predicted sense and alternatives
        """
        prep = triple.indicator.lower()
        traj = triple.trajector.lower()
        land = triple.landmark.lower()

        # Get possible senses for this preposition
        possible_senses = self.PREPOSITION_SENSES.get(prep, [PrepositionSense.GENERIC])

        if len(possible_senses) == 1:
            # Unambiguous preposition
            return SenseDisambiguation(
                preposition=prep,
                predicted_sense=possible_senses[0],
                confidence=1.0,
                alternatives={possible_senses[0]: 1.0},
                trajector_lemma=traj,
                landmark_lemma=land,
                is_functional=self._is_functional_sense(possible_senses[0]),
            )

        # Check for landmark-specific triggers
        if land in self.LANDMARK_SENSE_TRIGGERS:
            triggers = self.LANDMARK_SENSE_TRIGGERS[land]
            if prep in triggers:
                triggered_sense = triggers[prep]
                return SenseDisambiguation(
                    preposition=prep,
                    predicted_sense=triggered_sense,
                    confidence=0.9,
                    alternatives=self._get_alternatives(triggered_sense, possible_senses),
                    trajector_lemma=traj,
                    landmark_lemma=land,
                    is_functional=self._is_functional_sense(triggered_sense),
                    support_type=self._get_support_type(triggered_sense),
                    containment_type=self._get_containment_type(triggered_sense),
                )

        # Use affordance-based disambiguation
        affordances = self.feature_extractor.AFFORDANCES.get(land, {})

        if prep == 'on':
            sense = self._disambiguate_on(affordances, land, traj)
        elif prep == 'in':
            sense = self._disambiguate_in(affordances, land, traj)
        else:
            sense = possible_senses[0] if possible_senses else PrepositionSense.GENERIC

        return SenseDisambiguation(
            preposition=prep,
            predicted_sense=sense,
            confidence=0.7,  # Lower confidence for non-triggered disambiguation
            alternatives=self._get_alternatives(sense, possible_senses),
            trajector_lemma=traj,
            landmark_lemma=land,
            is_functional=self._is_functional_sense(sense),
            support_type=self._get_support_type(sense),
            containment_type=self._get_containment_type(sense),
        )

    def disambiguate_batch(self, triples: List[SpatialTriple],
                          tokens_list: Optional[List[List[Token]]] = None) -> List[SenseDisambiguation]:
        """
        Disambiguate multiple spatial triples.

        Args:
            triples: List of spatial triples
            tokens_list: Optional list of token lists

        Returns:
            List of disambiguations
        """
        results = []
        for i, triple in enumerate(triples):
            tokens = tokens_list[i] if tokens_list else None
            results.append(self.disambiguate(triple, tokens))
        return results

    def _disambiguate_on(self, affordances: Dict, landmark: str, trajector: str) -> PrepositionSense:
        """Disambiguate 'on' based on landmark properties."""
        orientation = affordances.get('orientation', 'horizontal')
        support_type = affordances.get('type', 'surface')

        if orientation == 'vertical':
            return PrepositionSense.ON_VERTICAL
        elif support_type == 'hanging':
            return PrepositionSense.ON_HANGING
        else:
            return PrepositionSense.ON_SUPPORT

    def _disambiguate_in(self, affordances: Dict, landmark: str, trajector: str) -> PrepositionSense:
        """Disambiguate 'in' based on landmark properties."""
        land_type = affordances.get('type', 'object')

        if land_type == 'region':
            return PrepositionSense.IN_REGION
        elif affordances.get('containment', False):
            return PrepositionSense.IN_CONTAINER
        else:
            # Default to functional containment for ambiguous cases
            return PrepositionSense.IN_FUNCTIONAL

    def _is_functional_sense(self, sense: PrepositionSense) -> bool:
        """Check if a sense involves functional (vs purely geometric) reasoning."""
        functional_senses = {
            PrepositionSense.ON_SUPPORT,
            PrepositionSense.ON_VERTICAL,
            PrepositionSense.ON_HANGING,
            PrepositionSense.IN_CONTAINER,
            PrepositionSense.IN_FUNCTIONAL,
        }
        return sense in functional_senses

    def _get_support_type(self, sense: PrepositionSense) -> Optional[str]:
        """Get the support type for support senses."""
        support_types = {
            PrepositionSense.ON_SUPPORT: 'horizontal',
            PrepositionSense.ON_VERTICAL: 'vertical',
            PrepositionSense.ON_HANGING: 'hanging',
        }
        return support_types.get(sense)

    def _get_containment_type(self, sense: PrepositionSense) -> Optional[str]:
        """Get the containment type for containment senses."""
        containment_types = {
            PrepositionSense.IN_CONTAINER: 'physical',
            PrepositionSense.IN_REGION: 'regional',
            PrepositionSense.IN_FUNCTIONAL: 'functional',
        }
        return containment_types.get(sense)

    def _get_alternatives(self, primary: PrepositionSense,
                         possible: List[PrepositionSense]) -> Dict[PrepositionSense, float]:
        """Get probability distribution over alternative senses."""
        alternatives = {}
        remaining_prob = 0.2  # 80% to primary, 20% distributed

        for sense in possible:
            if sense == primary:
                alternatives[sense] = 0.8
            else:
                alternatives[sense] = remaining_prob / (len(possible) - 1) if len(possible) > 1 else 0

        return alternatives

    def get_gum_modality(self, sense: PrepositionSense) -> str:
        """
        Map preposition sense to GUM SpatialModality category.

        Returns the GUM ontology class that corresponds to this sense.
        """
        gum_mapping = {
            # Support senses -> GUM Support/Control
            PrepositionSense.ON_SUPPORT: 'Support',
            PrepositionSense.ON_VERTICAL: 'Support',  # Subtype
            PrepositionSense.ON_HANGING: 'Support',   # Subtype

            # Containment senses -> GUM Containment/Control
            PrepositionSense.IN_CONTAINER: 'Containment',
            PrepositionSense.IN_REGION: 'Parthood',  # Regional containment
            PrepositionSense.IN_FUNCTIONAL: 'Containment',  # Functional

            # Location senses -> GUM Proximity
            PrepositionSense.AT_LOCATION: 'Proximity',
            PrepositionSense.NEAR_PROXIMITY: 'Proximity',
            PrepositionSense.BY_ADJACENCY: 'Proximity',

            # Projective senses -> GUM Projection
            PrepositionSense.ABOVE_VERTICAL: 'VerticalProjection',
            PrepositionSense.BELOW_VERTICAL: 'VerticalProjection',
            PrepositionSense.FRONT_PROJECTIVE: 'FrontalProjection',
            PrepositionSense.BEHIND_PROJECTIVE: 'FrontalProjection',

            # Default
            PrepositionSense.GENERIC: 'SpatialModality',
            PrepositionSense.UNKNOWN: 'SpatialModality',
        }
        return gum_mapping.get(sense, 'SpatialModality')


def demonstrate_disambiguation():
    """Demonstrate preposition sense disambiguation."""
    print("Preposition Sense Disambiguation Demonstration")
    print("=" * 50)

    disambiguator = PrepositionDisambiguator()

    # Test cases showing functional variation
    test_cases = [
        SpatialTriple("cup", "on", "table"),
        SpatialTriple("picture", "on", "wall"),
        SpatialTriple("coat", "on", "hook"),
        SpatialTriple("flowers", "in", "vase"),
        SpatialTriple("cat", "in", "room"),
        SpatialTriple("fish", "in", "net"),
        SpatialTriple("book", "on", "shelf"),
        SpatialTriple("poster", "on", "door"),
    ]

    print("\nDisambiguation Results:")
    print("-" * 50)

    for triple in test_cases:
        result = disambiguator.disambiguate(triple)

        print(f"\n'{triple.trajector} {triple.indicator} {triple.landmark}'")
        print(f"  Sense: {result.predicted_sense.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Is Functional: {result.is_functional}")
        if result.support_type:
            print(f"  Support Type: {result.support_type}")
        if result.containment_type:
            print(f"  Containment Type: {result.containment_type}")
        print(f"  GUM Modality: {disambiguator.get_gum_modality(result.predicted_sense)}")


if __name__ == "__main__":
    demonstrate_disambiguation()
