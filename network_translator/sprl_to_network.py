"""
SpRL to Network Translator

Converts SpRL (Spatial Role Labeling) outputs directly into
FA-QSR constraint networks without requiring explicit GUM annotation.

This provides a streamlined NLP-to-reasoning pipeline:
    Text -> SpRL -> FA-QSR Network -> Reasoning

The module handles:
- Entity extraction and variable creation
- Constraint generation from spatial triples
- Network consistency checking during construction
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsr_base.rcc8 import RCC8Relation, RelationSet, relation_set, UNIVERSAL
from qsr_base.faqsr_calculus import (
    FAQSRNetwork,
    FAQSRConstraint,
    FunctionalRelation,
)
from nlp_models.sprl_model import SpatialTriple, SpatialRoleLabeler
from nlp_models.preposition_model import PrepositionDisambiguator, PrepositionSense
from feature_engineering.spatial_features import Token


@dataclass
class EntityInfo:
    """Information about a spatial entity extracted from text."""
    name: str                    # Entity name/identifier
    mention: str                 # Original text mention
    entity_type: str = "object"  # Type: object, surface, container, region
    affordances: Set[str] = field(default_factory=set)
    sentence_idx: int = 0        # Index of sentence where mentioned


@dataclass
class NetworkBuildResult:
    """Result of building an FA-QSR network from text."""
    network: FAQSRNetwork
    entities: Dict[str, EntityInfo]
    triples: List[SpatialTriple]
    constraints: List[FAQSRConstraint]
    is_consistent: bool
    warnings: List[str] = field(default_factory=list)


class NetworkBuilder:
    """
    Builds FA-QSR networks from spatial text.

    Handles entity resolution, constraint generation, and
    consistency checking during network construction.
    """

    # Default affordance mappings
    DEFAULT_AFFORDANCES = {
        'table': {'support'},
        'desk': {'support'},
        'shelf': {'support'},
        'floor': {'support'},
        'wall': {'support', 'vertical'},
        'hook': {'support', 'hanging'},
        'box': {'containment'},
        'bag': {'containment'},
        'cup': {'containment'},
        'bowl': {'containment'},
        'vase': {'containment', 'support'},
        'room': {'containment', 'region'},
        'house': {'containment', 'region'},
        'city': {'containment', 'region'},
    }

    def __init__(self):
        self.disambiguator = PrepositionDisambiguator()
        self._entity_counter = 0

    def build_from_triples(self, triples: List[SpatialTriple]) -> NetworkBuildResult:
        """
        Build an FA-QSR network from spatial triples.

        Args:
            triples: List of spatial triples

        Returns:
            NetworkBuildResult with network and metadata
        """
        network = FAQSRNetwork()
        entities: Dict[str, EntityInfo] = {}
        constraints: List[FAQSRConstraint] = []
        warnings: List[str] = []

        for triple in triples:
            # Extract/create entities
            traj_entity = self._get_or_create_entity(triple.trajector, entities)
            land_entity = self._get_or_create_entity(triple.landmark, entities)

            # Add variables to network with affordance info
            network.add_variable(
                traj_entity.name,
                entity_type=traj_entity.entity_type,
                affordances=traj_entity.affordances
            )
            network.add_variable(
                land_entity.name,
                entity_type=land_entity.entity_type,
                affordances=land_entity.affordances
            )

            # Generate constraint from triple
            constraint = self._triple_to_constraint(triple, traj_entity, land_entity)
            constraints.append(constraint)

            # Add to network
            success = network.add_constraint(constraint)
            if not success:
                warnings.append(f"Constraint conflict: {constraint}")

        # Check consistency
        is_consistent = network.is_trivially_consistent() if hasattr(network, 'is_trivially_consistent') else True

        return NetworkBuildResult(
            network=network,
            entities=entities,
            triples=triples,
            constraints=constraints,
            is_consistent=is_consistent,
            warnings=warnings
        )

    def _get_or_create_entity(self, mention: str,
                              entities: Dict[str, EntityInfo]) -> EntityInfo:
        """Get existing entity or create new one from mention."""
        # Normalize mention
        normalized = mention.lower().strip()

        # Check if entity exists
        for name, info in entities.items():
            if info.mention.lower() == normalized:
                return info

        # Create new entity
        entity_name = self._generate_entity_name(normalized)

        # Determine type and affordances
        entity_type = self._infer_entity_type(normalized)
        affordances = self.DEFAULT_AFFORDANCES.get(normalized, set())

        entity = EntityInfo(
            name=entity_name,
            mention=mention,
            entity_type=entity_type,
            affordances=affordances
        )

        entities[entity_name] = entity
        return entity

    def _generate_entity_name(self, mention: str) -> str:
        """Generate unique entity name."""
        # Clean the mention for use as variable name
        clean = ''.join(c if c.isalnum() else '_' for c in mention)
        self._entity_counter += 1
        return f"{clean}_{self._entity_counter}"

    def _infer_entity_type(self, mention: str) -> str:
        """Infer entity type from mention."""
        containers = {'box', 'bag', 'cup', 'bowl', 'vase', 'basket', 'drawer'}
        surfaces = {'table', 'desk', 'shelf', 'floor', 'counter', 'ground'}
        vertical = {'wall', 'door', 'board', 'ceiling'}
        regions = {'room', 'house', 'building', 'city', 'country', 'area'}
        hangers = {'hook', 'peg', 'hanger', 'nail', 'branch'}

        if mention in containers:
            return 'container'
        elif mention in surfaces:
            return 'surface'
        elif mention in vertical:
            return 'vertical_surface'
        elif mention in regions:
            return 'region'
        elif mention in hangers:
            return 'hanger'
        else:
            return 'object'

    def _triple_to_constraint(self, triple: SpatialTriple,
                              traj: EntityInfo,
                              land: EntityInfo) -> FAQSRConstraint:
        """Convert spatial triple to FA-QSR constraint."""
        # Disambiguate preposition
        disambiguation = self.disambiguator.disambiguate(triple)
        sense = disambiguation.predicted_sense

        # Map sense to geometric constraints
        geometric = self._sense_to_geometric(sense)

        # Map sense to functional constraints
        functional = self._sense_to_functional(sense, land.entity_type)

        return FAQSRConstraint(
            var1=traj.name,
            var2=land.name,
            geometric=geometric,
            functional=functional
        )

    def _sense_to_geometric(self, sense: PrepositionSense) -> RelationSet:
        """Map preposition sense to RCC-8 relations."""
        DC = RCC8Relation.DC
        EC = RCC8Relation.EC
        PO = RCC8Relation.PO
        TPP = RCC8Relation.TPP
        NTPP = RCC8Relation.NTPP
        TPPi = RCC8Relation.TPPi
        NTPPi = RCC8Relation.NTPPi

        sense_to_geo = {
            # Support senses - contact
            PrepositionSense.ON_SUPPORT: relation_set(EC, PO),
            PrepositionSense.ON_VERTICAL: relation_set(EC, PO),
            PrepositionSense.ON_HANGING: relation_set(EC, PO),

            # Containment senses - inside
            PrepositionSense.IN_CONTAINER: relation_set(TPP, NTPP, PO),
            PrepositionSense.IN_FUNCTIONAL: relation_set(TPP, NTPP, PO),
            PrepositionSense.IN_REGION: relation_set(TPP, NTPP),

            # Proximity senses - disjoint or contact
            PrepositionSense.AT_LOCATION: relation_set(DC, EC, PO),
            PrepositionSense.NEAR_PROXIMITY: relation_set(DC, EC),
            PrepositionSense.BY_ADJACENCY: relation_set(DC, EC),

            # Projective senses - typically disjoint
            PrepositionSense.ABOVE_VERTICAL: relation_set(DC, EC),
            PrepositionSense.BELOW_VERTICAL: relation_set(DC, EC),
            PrepositionSense.FRONT_PROJECTIVE: relation_set(DC, EC),
            PrepositionSense.BEHIND_PROJECTIVE: relation_set(DC, EC),

            # Default
            PrepositionSense.GENERIC: UNIVERSAL,
            PrepositionSense.UNKNOWN: UNIVERSAL,
        }

        return sense_to_geo.get(sense, UNIVERSAL)

    def _sense_to_functional(self, sense: PrepositionSense,
                            landmark_type: str) -> frozenset:
        """Map preposition sense to functional relations."""
        sense_to_func = {
            PrepositionSense.ON_SUPPORT: frozenset([FunctionalRelation.FSUPPORT]),
            PrepositionSense.ON_VERTICAL: frozenset([FunctionalRelation.FSUPPORT_ADHERE]),
            PrepositionSense.ON_HANGING: frozenset([FunctionalRelation.FSUPPORT_HANG]),
            PrepositionSense.IN_CONTAINER: frozenset([FunctionalRelation.FCONTAIN_PARTIAL]),
            PrepositionSense.IN_FUNCTIONAL: frozenset([FunctionalRelation.FCONTAIN]),
            PrepositionSense.IN_REGION: frozenset([FunctionalRelation.FCONTAIN_BOUNDARY]),
        }

        return sense_to_func.get(sense, frozenset())


class SpRLToNetworkTranslator:
    """
    End-to-end translator from text to FA-QSR network.

    Combines SpRL for role labeling with network building.
    """

    def __init__(self):
        self.labeler = SpatialRoleLabeler()
        self.builder = NetworkBuilder()

    def translate(self, tokens: List[Token]) -> NetworkBuildResult:
        """
        Translate tokenized sentence to FA-QSR network.

        Args:
            tokens: Tokenized sentence with POS/dependency annotations

        Returns:
            NetworkBuildResult with FA-QSR network
        """
        # Extract spatial triples
        triples = self.labeler.extract_triples(tokens)

        if not triples:
            # Return empty network
            return NetworkBuildResult(
                network=FAQSRNetwork(),
                entities={},
                triples=[],
                constraints=[],
                is_consistent=True,
                warnings=["No spatial expressions found in input"]
            )

        # Build network from triples
        return self.builder.build_from_triples(triples)

    def translate_text(self, text: str) -> NetworkBuildResult:
        """
        Translate raw text to FA-QSR network.

        Note: This is a simplified implementation. Full implementation
        would use a proper NLP pipeline (spaCy, etc.) for tokenization
        and parsing.

        Args:
            text: Raw text string

        Returns:
            NetworkBuildResult with FA-QSR network
        """
        # Simple tokenization (would use spaCy in production)
        tokens = self._simple_tokenize(text)

        return self.translate(tokens)

    def _simple_tokenize(self, text: str) -> List[Token]:
        """
        Simple rule-based tokenization for demonstration.

        In production, use spaCy or similar for proper parsing.
        """
        import re

        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)

        tokens = []
        for i, word in enumerate(words):
            # Simple POS heuristics
            pos = self._guess_pos(word)

            # Simple dependency (everything points to previous word)
            head_idx = max(0, i - 1)

            # Guess dependency relation
            dep = self._guess_dep(word, pos)

            token = Token(
                text=word,
                lemma=word.lower(),
                pos=pos,
                dep=dep,
                head_idx=head_idx,
                idx=i
            )
            tokens.append(token)

        return tokens

    def _guess_pos(self, word: str) -> str:
        """Simple POS guessing."""
        word_lower = word.lower()

        prepositions = {'in', 'on', 'at', 'by', 'near', 'under', 'over', 'above', 'below', 'behind'}
        determiners = {'the', 'a', 'an', 'this', 'that'}
        verbs = {'is', 'are', 'was', 'were', 'be', 'put', 'place', 'sit', 'stand'}

        if word_lower in prepositions:
            return 'IN'
        elif word_lower in determiners:
            return 'DT'
        elif word_lower in verbs:
            return 'VBZ'
        elif word[0].isupper():
            return 'NNP'
        else:
            return 'NN'

    def _guess_dep(self, word: str, pos: str) -> str:
        """Simple dependency relation guessing."""
        if pos == 'IN':
            return 'prep'
        elif pos == 'DT':
            return 'det'
        elif pos == 'NN' or pos == 'NNP':
            return 'pobj'  # Default assumption
        else:
            return 'ROOT'


def demonstrate_sprl_translator():
    """Demonstrate SpRL to Network translation."""
    print("SpRL to FA-QSR Network Translation")
    print("=" * 50)

    translator = SpRLToNetworkTranslator()

    # Test sentences
    test_sentences = [
        "The book is on the table",
        "The flowers are in the vase on the shelf",
        "The picture hangs on the wall above the desk",
    ]

    for sentence in test_sentences:
        print(f"\nInput: '{sentence}'")
        print("-" * 40)

        result = translator.translate_text(sentence)

        print(f"Entities found: {len(result.entities)}")
        for name, info in result.entities.items():
            print(f"  {name}: {info.mention} (type: {info.entity_type})")

        print(f"\nTriples extracted: {len(result.triples)}")
        for triple in result.triples:
            print(f"  {triple}")

        print(f"\nConstraints generated: {len(result.constraints)}")
        for constraint in result.constraints:
            print(f"  {constraint}")

        print(f"\nNetwork consistent: {result.is_consistent}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")


if __name__ == "__main__":
    demonstrate_sprl_translator()
