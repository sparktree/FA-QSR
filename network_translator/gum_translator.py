"""
GUM to FA-QSR Translator

This module implements the deterministic translation bridge from GUM
(Generalized Upper Model) spatial modalities to FA-QSR constraint sets.

The translator maps:
- GUM SpatialLocating configurations -> FA-QSR constraints
- GUM SpatialModality classes -> Geometric + Functional constraints
- GUM Projection types -> RCC-8 + directional constraints
- GUM FunctionalSpatialModality -> FA-QSR functional primitives

This is Phase 2 of the FA-QSR framework as described in the methodology.

References:
- Bateman et al. (2010) - GUM 3.0 Spatial Extension
- Cohn & Renz (2008) - Qualitative Spatial Representation and Reasoning
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsr_base.rcc8 import (
    RCC8Relation,
    RelationSet,
    relation_set,
    UNIVERSAL,
)
from qsr_base.faqsr_calculus import (
    FAQSRNetwork,
    FAQSRConstraint,
    FunctionalRelation,
    FunctionalRelationSet,
)
from nlp_models.sprl_model import SpatialTriple
from nlp_models.preposition_model import PrepositionSense, PrepositionDisambiguator


class GUMSpatialModality(Enum):
    """
    GUM SpatialModality classes relevant to FA-QSR translation.

    Based on GUM-3-space.owl ontology structure.
    """
    # Abstract classes
    SPATIAL_MODALITY = "SpatialModality"
    RELATIVE_SPATIAL_MODALITY = "RelativeSpatialModality"

    # Topological/Connection
    CONNECTION = "Connection"
    PARTHOOD = "Parthood"
    DISJOINTNESS = "Disjointness"
    PROXIMITY = "Proximity"

    # Functional modalities
    FUNCTIONAL_SPATIAL_MODALITY = "FunctionalSpatialModality"
    CONTROL = "Control"
    SUPPORT = "Support"
    CONTAINMENT = "Containment"
    ACCESS = "Access"
    DENIAL_OF_FUNCTIONAL_CONTROL = "DenialOfFunctionalControl"

    # Projective modalities
    PROJECTION = "Projection"
    VERTICAL_PROJECTION = "VerticalProjection"
    ABOVE_PROJECTION = "AboveProjection"
    BELOW_PROJECTION = "BelowProjection"
    FRONTAL_PROJECTION = "FrontalProjection"
    FRONT_PROJECTION = "FrontProjection"
    BACK_PROJECTION = "BackProjection"
    LATERAL_PROJECTION = "LateralProjection"
    LEFT_PROJECTION = "LeftProjection"
    RIGHT_PROJECTION = "RightProjection"

    # Internal/External variants
    ABOVE_PROJECTION_EXTERNAL = "AboveProjectionExternal"
    ABOVE_PROJECTION_INTERNAL = "AboveProjectionInternal"
    BELOW_PROJECTION_EXTERNAL = "BelowProjectionExternal"
    BELOW_PROJECTION_INTERNAL = "BelowProjectionInternal"


@dataclass
class TranslationConfig:
    """Configuration for GUM to FA-QSR translation."""
    # Whether to include functional constraints
    include_functional: bool = True

    # Whether to apply default axioms
    apply_defaults: bool = True

    # Strictness of geometric requirements
    strict_geometric: bool = False

    # Include complexity annotations
    annotate_complexity: bool = True


@dataclass
class TranslationResult:
    """Result of translating a spatial expression to FA-QSR."""
    # The generated constraint
    constraint: FAQSRConstraint

    # Source GUM modality
    gum_modality: GUMSpatialModality

    # Complexity class
    complexity_class: str  # 'tractable', 'np-hard', 'unknown'

    # Translation confidence
    confidence: float

    # Warnings or notes
    notes: List[str] = field(default_factory=list)


class GUMToFAQSRTranslator:
    """
    Translates GUM spatial modalities to FA-QSR constraint networks.

    Implements the semantic bridge between linguistically-motivated
    GUM categories and the algebraic machinery of FA-QSR.
    """

    def __init__(self, config: Optional[TranslationConfig] = None):
        """
        Initialize the translator.

        Args:
            config: Translation configuration options
        """
        self.config = config or TranslationConfig()
        self.disambiguator = PrepositionDisambiguator()

        # Build translation tables
        self._build_modality_to_geometric_map()
        self._build_modality_to_functional_map()
        self._build_sense_to_modality_map()

    def _build_modality_to_geometric_map(self):
        """
        Build mapping from GUM modalities to RCC-8 relation sets.

        These represent the geometric requirements/implications of each modality.
        """
        DC = RCC8Relation.DC
        EC = RCC8Relation.EC
        PO = RCC8Relation.PO
        EQ = RCC8Relation.EQ
        TPP = RCC8Relation.TPP
        NTPP = RCC8Relation.NTPP
        TPPi = RCC8Relation.TPPi
        NTPPi = RCC8Relation.NTPPi

        self._geo_map: Dict[GUMSpatialModality, RelationSet] = {
            # Topological modalities
            GUMSpatialModality.CONNECTION: relation_set(EC, PO, TPP, NTPP, TPPi, NTPPi, EQ),
            GUMSpatialModality.PARTHOOD: relation_set(TPP, NTPP, EQ),
            GUMSpatialModality.DISJOINTNESS: relation_set(DC),
            GUMSpatialModality.PROXIMITY: relation_set(DC, EC),

            # Functional modalities - have geometric prerequisites
            GUMSpatialModality.SUPPORT: relation_set(EC, PO),
            GUMSpatialModality.CONTAINMENT: relation_set(PO, TPPi, NTPPi),
            GUMSpatialModality.CONTROL: relation_set(EC, PO, TPP, NTPP, TPPi, NTPPi),
            GUMSpatialModality.ACCESS: relation_set(EC, PO, TPP, NTPP),

            # Projective modalities - external (disjoint) variants
            GUMSpatialModality.ABOVE_PROJECTION_EXTERNAL: relation_set(DC, EC),
            GUMSpatialModality.BELOW_PROJECTION_EXTERNAL: relation_set(DC, EC),

            # Projective modalities - internal (part) variants
            GUMSpatialModality.ABOVE_PROJECTION_INTERNAL: relation_set(TPP, NTPP),
            GUMSpatialModality.BELOW_PROJECTION_INTERNAL: relation_set(TPP, NTPP),

            # Abstract/underspecified - allow anything
            GUMSpatialModality.SPATIAL_MODALITY: UNIVERSAL,
            GUMSpatialModality.RELATIVE_SPATIAL_MODALITY: UNIVERSAL,
            GUMSpatialModality.PROJECTION: UNIVERSAL,
        }

    def _build_modality_to_functional_map(self):
        """
        Build mapping from GUM modalities to FA-QSR functional relations.
        """
        self._func_map: Dict[GUMSpatialModality, FunctionalRelationSet] = {
            # Support modality -> fsupport
            GUMSpatialModality.SUPPORT: frozenset([FunctionalRelation.FSUPPORT]),

            # Containment modality -> fcontainment
            GUMSpatialModality.CONTAINMENT: frozenset([FunctionalRelation.FCONTAIN]),

            # Control is more general
            GUMSpatialModality.CONTROL: frozenset([
                FunctionalRelation.FSUPPORT,
                FunctionalRelation.FCONTAIN
            ]),

            # Access
            GUMSpatialModality.ACCESS: frozenset([FunctionalRelation.FACCESS]),

            # Denial of control
            GUMSpatialModality.DENIAL_OF_FUNCTIONAL_CONTROL: frozenset([
                FunctionalRelation.NO_FSUPPORT,
                FunctionalRelation.NO_FCONTAIN
            ]),

            # Projective and topological modalities don't have functional component
            GUMSpatialModality.ABOVE_PROJECTION_EXTERNAL: frozenset(),
            GUMSpatialModality.BELOW_PROJECTION_EXTERNAL: frozenset(),
            GUMSpatialModality.PARTHOOD: frozenset(),
            GUMSpatialModality.DISJOINTNESS: frozenset(),
            GUMSpatialModality.PROXIMITY: frozenset(),
        }

    def _build_sense_to_modality_map(self):
        """
        Map preposition senses to GUM modalities.
        """
        self._sense_to_modality: Dict[PrepositionSense, GUMSpatialModality] = {
            # Support senses
            PrepositionSense.ON_SUPPORT: GUMSpatialModality.SUPPORT,
            PrepositionSense.ON_VERTICAL: GUMSpatialModality.SUPPORT,
            PrepositionSense.ON_HANGING: GUMSpatialModality.SUPPORT,

            # Containment senses
            PrepositionSense.IN_CONTAINER: GUMSpatialModality.CONTAINMENT,
            PrepositionSense.IN_FUNCTIONAL: GUMSpatialModality.CONTAINMENT,
            PrepositionSense.IN_REGION: GUMSpatialModality.PARTHOOD,

            # Proximity senses
            PrepositionSense.AT_LOCATION: GUMSpatialModality.PROXIMITY,
            PrepositionSense.NEAR_PROXIMITY: GUMSpatialModality.PROXIMITY,
            PrepositionSense.BY_ADJACENCY: GUMSpatialModality.PROXIMITY,

            # Projective senses
            PrepositionSense.ABOVE_VERTICAL: GUMSpatialModality.ABOVE_PROJECTION_EXTERNAL,
            PrepositionSense.BELOW_VERTICAL: GUMSpatialModality.BELOW_PROJECTION_EXTERNAL,
            PrepositionSense.FRONT_PROJECTIVE: GUMSpatialModality.FRONT_PROJECTION,
            PrepositionSense.BEHIND_PROJECTIVE: GUMSpatialModality.BACK_PROJECTION,

            # Default
            PrepositionSense.GENERIC: GUMSpatialModality.SPATIAL_MODALITY,
            PrepositionSense.UNKNOWN: GUMSpatialModality.SPATIAL_MODALITY,
        }

    def translate_triple(self, triple: SpatialTriple) -> TranslationResult:
        """
        Translate a spatial triple to FA-QSR constraint.

        This is the primary translation interface.

        Args:
            triple: Spatial triple from SpRL

        Returns:
            TranslationResult with FA-QSR constraint
        """
        # Step 1: Disambiguate preposition sense
        disambiguation = self.disambiguator.disambiguate(triple)
        sense = disambiguation.predicted_sense

        # Step 2: Map sense to GUM modality
        modality = self._sense_to_modality.get(sense, GUMSpatialModality.SPATIAL_MODALITY)

        # Step 3: Get geometric constraints
        geometric = self._geo_map.get(modality, UNIVERSAL)

        # Step 4: Get functional constraints (if enabled)
        functional: FunctionalRelationSet = frozenset()
        if self.config.include_functional:
            functional = self._func_map.get(modality, frozenset())

            # Apply sense-specific refinements
            functional = self._refine_functional_for_sense(sense, functional)

        # Step 5: Create constraint
        constraint = FAQSRConstraint(
            var1=triple.trajector,
            var2=triple.landmark,
            geometric=geometric,
            functional=functional
        )

        # Step 6: Determine complexity class
        complexity = self._determine_complexity(constraint)

        # Step 7: Generate notes
        notes = self._generate_notes(triple, sense, modality)

        return TranslationResult(
            constraint=constraint,
            gum_modality=modality,
            complexity_class=complexity,
            confidence=disambiguation.confidence,
            notes=notes
        )

    def translate_modality(self, modality: GUMSpatialModality,
                          trajector: str, landmark: str) -> FAQSRConstraint:
        """
        Directly translate a GUM modality to FA-QSR constraint.

        Useful when GUM annotations are available directly.

        Args:
            modality: GUM spatial modality
            trajector: Trajector variable name
            landmark: Landmark variable name

        Returns:
            FA-QSR constraint
        """
        geometric = self._geo_map.get(modality, UNIVERSAL)

        functional: FunctionalRelationSet = frozenset()
        if self.config.include_functional:
            functional = self._func_map.get(modality, frozenset())

        return FAQSRConstraint(
            var1=trajector,
            var2=landmark,
            geometric=geometric,
            functional=functional
        )

    def build_network(self, triples: List[SpatialTriple]) -> Tuple[FAQSRNetwork, List[TranslationResult]]:
        """
        Build an FA-QSR network from multiple spatial triples.

        Args:
            triples: List of spatial triples

        Returns:
            Tuple of (FAQSRNetwork, list of TranslationResults)
        """
        network = FAQSRNetwork()
        results = []

        for triple in triples:
            # Translate triple
            result = self.translate_triple(triple)
            results.append(result)

            # Add to network
            constraint = result.constraint

            # Add variables
            network.add_variable(constraint.var1)
            network.add_variable(constraint.var2)

            # Add constraint
            network.add_constraint(constraint)

        return network, results

    def _refine_functional_for_sense(self, sense: PrepositionSense,
                                     functional: FunctionalRelationSet) -> FunctionalRelationSet:
        """
        Refine functional relations based on specific preposition sense.
        """
        # Map senses to specific functional subtypes
        sense_refinements = {
            PrepositionSense.ON_HANGING: frozenset([FunctionalRelation.FSUPPORT_HANG]),
            PrepositionSense.ON_VERTICAL: frozenset([FunctionalRelation.FSUPPORT_ADHERE]),
            PrepositionSense.IN_CONTAINER: frozenset([FunctionalRelation.FCONTAIN_PARTIAL]),
        }

        return sense_refinements.get(sense, functional)

    def _determine_complexity(self, constraint: FAQSRConstraint) -> str:
        """
        Determine computational complexity class of constraint.
        """
        if not constraint.functional:
            # Pure geometric - likely tractable
            if len(constraint.geometric) <= 3:
                return 'tractable'
            else:
                return 'unknown'
        else:
            # Has functional component - likely NP-hard when combined
            return 'np-hard'

    def _generate_notes(self, triple: SpatialTriple,
                       sense: PrepositionSense,
                       modality: GUMSpatialModality) -> List[str]:
        """Generate translation notes/warnings."""
        notes = []

        # Note functional interpretations
        if modality in (GUMSpatialModality.SUPPORT, GUMSpatialModality.CONTAINMENT):
            notes.append(f"Functional interpretation: {modality.value}")

        # Note ambiguity
        if sense in (PrepositionSense.GENERIC, PrepositionSense.UNKNOWN):
            notes.append("Preposition sense could not be fully disambiguated")

        return notes


class GUMOntologyInterface:
    """
    Interface for reading GUM ontology classes and properties.

    Provides methods to query the GUM-space.owl ontology for
    class hierarchies and constraints.
    """

    # GUM namespace
    GUM_NS = "http://www.ontospace.uni-bremen.de/ontology/stable/GUM-3.owl#"
    GUM_SPACE_NS = "http://www.ontospace.uni-bremen.de/ontology/stable/GUM-3-space.owl#"

    def __init__(self, ontology_path: Optional[str] = None):
        """
        Initialize the interface.

        Args:
            ontology_path: Path to GUM-space.owl file (optional)
        """
        self.ontology_path = ontology_path
        self._classes: Dict[str, Set[str]] = {}  # class -> superclasses
        self._properties: Dict[str, Dict[str, str]] = {}

        # Build class hierarchy from known GUM structure
        self._build_class_hierarchy()

    def _build_class_hierarchy(self):
        """Build class hierarchy from GUM ontology knowledge."""
        self._classes = {
            'SpatialModality': set(),
            'RelativeSpatialModality': {'SpatialModality'},
            'FunctionalSpatialModality': {'SpatialModality'},

            # Topological
            'Connection': {'RelativeSpatialModality'},
            'Parthood': {'RelativeSpatialModality'},
            'Disjointness': {'RelativeSpatialModality'},
            'Proximity': {'RelativeSpatialModality'},

            # Functional
            'Control': {'FunctionalSpatialModality'},
            'Support': {'Control'},
            'Containment': {'Control'},
            'Access': {'FunctionalSpatialModality'},

            # Projective
            'Projection': {'SpatialModality'},
            'VerticalProjection': {'Projection'},
            'AboveProjection': {'VerticalProjection'},
            'BelowProjection': {'VerticalProjection'},
            'FrontalProjection': {'Projection'},
            'FrontProjection': {'FrontalProjection'},
            'BackProjection': {'FrontalProjection'},
        }

    def get_superclasses(self, class_name: str) -> Set[str]:
        """Get all superclasses of a GUM class."""
        if class_name not in self._classes:
            return set()

        result = set()
        direct = self._classes[class_name]
        result.update(direct)

        for superclass in direct:
            result.update(self.get_superclasses(superclass))

        return result

    def is_subclass_of(self, class_name: str, superclass: str) -> bool:
        """Check if one class is a subclass of another."""
        if class_name == superclass:
            return True
        return superclass in self.get_superclasses(class_name)

    def is_functional(self, class_name: str) -> bool:
        """Check if a modality class is functional (vs purely geometric)."""
        return self.is_subclass_of(class_name, 'FunctionalSpatialModality')

    def is_projective(self, class_name: str) -> bool:
        """Check if a modality class is projective."""
        return self.is_subclass_of(class_name, 'Projection')


def demonstrate_translator():
    """Demonstrate the GUM to FA-QSR translator."""
    print("GUM to FA-QSR Translation Demonstration")
    print("=" * 50)

    translator = GUMToFAQSRTranslator()

    # Test spatial triples
    test_triples = [
        SpatialTriple("cup", "on", "table"),
        SpatialTriple("picture", "on", "wall"),
        SpatialTriple("coat", "on", "hook"),
        SpatialTriple("flowers", "in", "vase"),
        SpatialTriple("cat", "in", "room"),
        SpatialTriple("book", "above", "desk"),
        SpatialTriple("dog", "near", "house"),
    ]

    print("\nTranslation Results:")
    print("-" * 50)

    for triple in test_triples:
        result = translator.translate_triple(triple)

        print(f"\n'{triple.trajector} {triple.indicator} {triple.landmark}'")
        print(f"  GUM Modality: {result.gum_modality.value}")
        print(f"  Geometric: {{{', '.join(str(r) for r in result.constraint.geometric)}}}")
        if result.constraint.functional:
            print(f"  Functional: {{{', '.join(str(r) for r in result.constraint.functional)}}}")
        print(f"  Complexity: {result.complexity_class}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.notes:
            print(f"  Notes: {result.notes}")

    # Build network from all triples
    print("\n" + "=" * 50)
    print("Building FA-QSR Network:")
    print("-" * 50)

    network, all_results = translator.build_network(test_triples)
    print(network)

    # Show complexity analysis
    print("\nNetwork Complexity Analysis:")
    complexity_info = network.estimate_complexity()
    for key, value in complexity_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demonstrate_translator()
