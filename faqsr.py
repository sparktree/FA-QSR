"""
FA-QSR: Functionally-Augmented Qualitative Spatial Reasoning

Main orchestration module for the FA-QSR framework.

This module provides a unified interface for:
1. Processing natural language spatial descriptions
2. Translating to FA-QSR constraint networks
3. Performing hybrid geometric-functional reasoning
4. Analyzing computational complexity

The framework bridges linguistically motivated GUM categories with
algebraic QSR machinery, enabling context-aware spatial inference
from natural language.

Usage:
    from faqsr import FAQSR

    # Initialize the framework
    faqsr = FAQSR()

    # Process spatial text
    result = faqsr.process("The flowers are in the vase on the table")

    # Check consistency
    is_consistent = faqsr.is_consistent(result.network)

    # Get inferences
    inferences = faqsr.infer(result.network, "flowers", "table")

References:
- Bateman et al. (2010) - GUM Spatial Extension
- Cohn & Renz (2008) - Qualitative Spatial Reasoning
- Herskovits (1986) - Language and Spatial Cognition
"""

import sys
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Core QSR imports
from qsr_base.rcc8 import (
    RCC8Relation,
    RelationSet,
    relation_set,
    UNIVERSAL,
    ConstraintNetwork,
)
from qsr_base.faqsr_calculus import (
    FAQSRNetwork,
    FAQSRConstraint,
    FunctionalRelation,
    FunctionalRelationSet,
)

# Reasoning imports
from reasoning_engine.path_consistency import (
    PathConsistencyChecker,
    FAQSRReasoner,
    ConsistencyResult,
    ConsistencyStatus,
)
from reasoning_engine.backtrack import BacktrackingSolver, FAQSRBacktrackingSolver

# NLP imports
from nlp_models.sprl_model import SpatialRoleLabeler, SpatialTriple
from nlp_models.preposition_model import PrepositionDisambiguator, PrepositionSense

# Translator imports
from network_translator.gum_translator import (
    GUMToFAQSRTranslator,
    TranslationResult,
    GUMSpatialModality,
)
from network_translator.sprl_to_network import (
    SpRLToNetworkTranslator,
    NetworkBuilder,
    NetworkBuildResult,
)

# Feature extraction
from feature_engineering.spatial_features import (
    SpatialFeatureExtractor,
    Token,
)

# Complexity analysis
from complexity_analysis.complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityResult,
    ComplexityClass,
)


@dataclass
class FAQSRConfig:
    """Configuration for FA-QSR processing."""
    # NLP settings
    use_heuristic_parsing: bool = True
    use_semantic_features: bool = True

    # Translation settings
    include_functional_constraints: bool = True
    apply_default_axioms: bool = True

    # Reasoning settings
    use_path_consistency: bool = True
    max_reasoning_iterations: int = 10000
    enable_backtracking: bool = False  # For NP-hard fragments

    # Complexity settings
    analyze_complexity: bool = True
    warn_on_np_hard: bool = True


@dataclass
class ProcessingResult:
    """Result of processing spatial text."""
    # Input
    input_text: str

    # Extracted information
    triples: List[SpatialTriple]
    entities: Dict[str, Any]

    # Generated network
    network: FAQSRNetwork
    constraints: List[FAQSRConstraint]

    # Analysis
    consistency_result: Optional[ConsistencyResult] = None
    complexity_result: Optional[ComplexityResult] = None

    # Metadata
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result of spatial inference query."""
    # Query
    entity1: str
    entity2: str

    # Inferred relations
    geometric_relations: RelationSet
    functional_relations: FunctionalRelationSet

    # Confidence and notes
    confidence: float = 1.0
    derivation_path: List[str] = field(default_factory=list)


class FAQSR:
    """
    Main FA-QSR framework class.

    Provides unified interface for all FA-QSR functionality.
    """

    def __init__(self, config: Optional[FAQSRConfig] = None):
        """
        Initialize the FA-QSR framework.

        Args:
            config: Configuration options (uses defaults if None)
        """
        self.config = config or FAQSRConfig()

        # Initialize components
        self._init_nlp_components()
        self._init_translator()
        self._init_reasoner()
        self._init_analyzer()

    def _init_nlp_components(self):
        """Initialize NLP pipeline components."""
        self.labeler = SpatialRoleLabeler(
            use_heuristics=self.config.use_heuristic_parsing
        )
        self.disambiguator = PrepositionDisambiguator()
        self.feature_extractor = SpatialFeatureExtractor(
            use_semantic_features=self.config.use_semantic_features
        )

    def _init_translator(self):
        """Initialize GUM-FA-QSR translator."""
        self.translator = GUMToFAQSRTranslator()
        self.network_builder = NetworkBuilder()
        self.sprl_translator = SpRLToNetworkTranslator()

    def _init_reasoner(self):
        """Initialize reasoning engine."""
        self.pc_checker = PathConsistencyChecker(
            max_iterations=self.config.max_reasoning_iterations
        )
        self.reasoner = FAQSRReasoner(
            max_iterations=self.config.max_reasoning_iterations
        )
        if self.config.enable_backtracking:
            self.backtrack_solver = FAQSRBacktrackingSolver()

    def _init_analyzer(self):
        """Initialize complexity analyzer."""
        self.complexity_analyzer = ComplexityAnalyzer()

    def process(self, text: str) -> ProcessingResult:
        """
        Process natural language spatial text.

        This is the main entry point for FA-QSR processing.

        Args:
            text: Natural language text containing spatial descriptions

        Returns:
            ProcessingResult with network and analysis
        """
        import time
        start_time = time.time()

        warnings = []

        # Step 1: Tokenize and parse
        tokens = self._tokenize(text)

        # Step 2: Extract spatial triples
        triples = self.labeler.extract_triples(tokens)

        if not triples:
            warnings.append("No spatial expressions found in input")

        # Step 3: Build FA-QSR network
        build_result = self.network_builder.build_from_triples(triples)
        network = build_result.network
        warnings.extend(build_result.warnings)

        # Step 4: Check consistency (if configured)
        consistency_result = None
        if self.config.use_path_consistency and triples:
            consistency_result = self.reasoner.check_consistency(network)
            if consistency_result.status == ConsistencyStatus.INCONSISTENT:
                warnings.append(f"Network inconsistency detected: {consistency_result.conflict}")

        # Step 5: Analyze complexity (if configured)
        complexity_result = None
        if self.config.analyze_complexity and triples:
            complexity_result = self.complexity_analyzer.analyze(network)
            if self.config.warn_on_np_hard and complexity_result.complexity_class == ComplexityClass.NP_HARD:
                warnings.append("Network contains NP-hard constraints")

        elapsed_ms = (time.time() - start_time) * 1000

        return ProcessingResult(
            input_text=text,
            triples=triples,
            entities=build_result.entities,
            network=network,
            constraints=build_result.constraints,
            consistency_result=consistency_result,
            complexity_result=complexity_result,
            processing_time_ms=elapsed_ms,
            warnings=warnings
        )

    def process_triples(self, triples: List[SpatialTriple]) -> ProcessingResult:
        """
        Process pre-extracted spatial triples.

        Useful when SpRL has already been performed externally.

        Args:
            triples: List of spatial triples

        Returns:
            ProcessingResult with network and analysis
        """
        import time
        start_time = time.time()

        # Build network from triples
        build_result = self.network_builder.build_from_triples(triples)
        network = build_result.network

        # Check consistency
        consistency_result = None
        if self.config.use_path_consistency:
            consistency_result = self.reasoner.check_consistency(network)

        # Analyze complexity
        complexity_result = None
        if self.config.analyze_complexity:
            complexity_result = self.complexity_analyzer.analyze(network)

        elapsed_ms = (time.time() - start_time) * 1000

        return ProcessingResult(
            input_text="[from triples]",
            triples=triples,
            entities=build_result.entities,
            network=network,
            constraints=build_result.constraints,
            consistency_result=consistency_result,
            complexity_result=complexity_result,
            processing_time_ms=elapsed_ms,
            warnings=build_result.warnings
        )

    def is_consistent(self, network: FAQSRNetwork) -> bool:
        """
        Check if a network is consistent.

        Args:
            network: FA-QSR network to check

        Returns:
            True if consistent, False otherwise
        """
        result = self.reasoner.check_consistency(network)
        return result.status == ConsistencyStatus.CONSISTENT

    def infer(self, network: FAQSRNetwork,
             entity1: str, entity2: str) -> InferenceResult:
        """
        Infer spatial relations between two entities.

        Uses constraint propagation to derive implied relations.

        Args:
            network: FA-QSR network
            entity1: First entity name
            entity2: Second entity name

        Returns:
            InferenceResult with inferred relations
        """
        # Get inferred constraint
        constraint = self.reasoner.infer_relations(network, entity1, entity2)

        return InferenceResult(
            entity1=entity1,
            entity2=entity2,
            geometric_relations=constraint.geometric,
            functional_relations=constraint.functional,
            confidence=1.0 if constraint.geometric else 0.0
        )

    def get_complexity(self, network: FAQSRNetwork) -> ComplexityResult:
        """
        Analyze complexity of a network.

        Args:
            network: FA-QSR network

        Returns:
            ComplexityResult with analysis
        """
        return self.complexity_analyzer.analyze(network)

    def _tokenize(self, text: str) -> List[Token]:
        """
        Tokenize text with simple rule-based approach.

        In production, replace with spaCy or similar.
        """
        import re

        words = re.findall(r'\b\w+\b', text)
        tokens = []

        for i, word in enumerate(words):
            pos = self._guess_pos(word)
            head_idx = max(0, i - 1)
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

        prepositions = {'in', 'on', 'at', 'by', 'near', 'under', 'over',
                       'above', 'below', 'behind', 'beside', 'between'}
        determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
        verbs = {'is', 'are', 'was', 'were', 'be', 'put', 'place', 'sit',
                'stand', 'hang', 'lies', 'sits', 'stands'}

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
            return 'pobj'
        elif pos == 'VBZ':
            return 'ROOT'
        else:
            return 'dep'


def create_example_network() -> FAQSRNetwork:
    """Create an example FA-QSR network for demonstration."""
    network = FAQSRNetwork()

    # Add entities
    network.add_variable("flowers", entity_type="object")
    network.add_variable("vase", entity_type="container",
                        affordances={"containment", "support"})
    network.add_variable("table", entity_type="surface",
                        affordances={"support"})
    network.add_variable("room", entity_type="region")

    # Add constraints
    # Flowers in vase - functional containment
    network.add_functional_constraint(
        "flowers", "vase",
        frozenset([FunctionalRelation.FCONTAIN_PARTIAL])
    )

    # Vase on table - functional support
    network.add_functional_constraint(
        "vase", "table",
        frozenset([FunctionalRelation.FSUPPORT])
    )

    # Table in room - geometric containment
    network.add_geometric_constraint(
        "table", "room",
        relation_set(RCC8Relation.TPP, RCC8Relation.NTPP)
    )

    return network


def main():
    """Main demonstration of FA-QSR capabilities."""
    print("=" * 60)
    print("FA-QSR: Functionally-Augmented Qualitative Spatial Reasoning")
    print("=" * 60)

    # Initialize framework
    faqsr = FAQSR()

    # Example 1: Process natural language
    print("\n1. Natural Language Processing")
    print("-" * 40)

    text = "The flowers are in the vase on the table"
    print(f"Input: '{text}'")

    result = faqsr.process(text)

    print(f"\nExtracted {len(result.triples)} spatial triple(s):")
    for triple in result.triples:
        print(f"  {triple}")

    print(f"\nNetwork has {result.network.num_variables} entities")

    if result.consistency_result:
        print(f"Consistency: {result.consistency_result.status.value}")

    if result.complexity_result:
        print(f"Complexity: {result.complexity_result.complexity_class.value}")
        print(f"Cognitive difficulty: {result.complexity_result.predicted_processing_difficulty}")

    print(f"\nProcessing time: {result.processing_time_ms:.2f}ms")

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")

    # Example 2: Direct network construction
    print("\n2. Direct Network Construction")
    print("-" * 40)

    network = create_example_network()
    print(f"Created network with {network.num_variables} variables")

    # Check consistency
    is_consistent = faqsr.is_consistent(network)
    print(f"Network consistent: {is_consistent}")

    # Infer relation between flowers and table
    inference = faqsr.infer(network, "flowers", "table")
    print(f"\nInferred relations (flowers -> table):")
    print(f"  Geometric: {{{', '.join(str(r) for r in inference.geometric_relations)}}}")
    if inference.functional_relations:
        print(f"  Functional: {{{', '.join(str(r) for r in inference.functional_relations)}}}")

    # Complexity analysis
    complexity = faqsr.get_complexity(network)
    print(f"\nComplexity analysis:")
    print(f"  Class: {complexity.complexity_class.value}")
    print(f"  Fragment type: {complexity.fragment_type.value}")
    print(f"  Tractable: {complexity.is_tractable}")

    # Example 3: Pre-extracted triples
    print("\n3. Processing Pre-extracted Triples")
    print("-" * 40)

    triples = [
        SpatialTriple("cup", "on", "saucer"),
        SpatialTriple("saucer", "on", "table"),
        SpatialTriple("table", "in", "kitchen"),
    ]

    result2 = faqsr.process_triples(triples)
    print(f"Processed {len(triples)} triples")
    print(f"Network consistent: {result2.consistency_result.status.value if result2.consistency_result else 'N/A'}")

    print("\n" + "=" * 60)
    print("FA-QSR Demonstration Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
