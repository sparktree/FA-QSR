"""
Empirical Analysis for FA-QSR Framework

This module implements systematic experiments to validate the theoretical claims
of the FA-QSR framework. It produces quantitative results for:

1. Preposition Sense Classification (Geometric vs Functional)
2. Constraint Fragment Complexity Analysis
3. Consistency Checking Performance
4. Translation Pipeline Accuracy
5. Cognitive Difficulty Correlation

The results support the claims in the accompanying paper:
"Integrating Formal Ontology with Qualitative Spatial Reasoning"

References:
- Landau (2024) - Geometric vs Functional spatial terms
- Renz & Nebel (1999) - Tractable fragments of RCC-8
- Coventry et al. (2001) - Functional geometry of prepositions
"""

import sys
import os
import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsr_base.rcc8 import (
    RCC8Relation,
    RelationSet,
    relation_set,
    UNIVERSAL,
    ConstraintNetwork,
    TractableFragments,
)
from qsr_base.faqsr_calculus import (
    FAQSRNetwork,
    FAQSRConstraint,
    FunctionalRelation,
)
from reasoning_engine.path_consistency import (
    PathConsistencyChecker,
    FAQSRReasoner,
    ConsistencyStatus,
)
from complexity_analysis.complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityClass,
    FragmentType,
)
from nlp_models.sprl_model import SpatialRoleLabeler, SpatialTriple
from nlp_models.preposition_model import PrepositionDisambiguator, PrepositionSense
from network_translator.gum_translator import GUMToFAQSRTranslator


class PrepositionCategory(Enum):
    """Preposition categories based on Landau (2024)."""
    GEOMETRIC = "geometric"      # above, below, behind, in front of
    FUNCTIONAL = "functional"    # on (support), in (containment)
    HYBRID = "hybrid"           # near, at (context-dependent)


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    experiment_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    raw_data: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CorpusStatistics:
    """Statistics from corpus analysis."""
    total_sentences: int
    total_triples: int
    preposition_counts: Dict[str, int]
    sense_distribution: Dict[str, int]
    functional_ratio: float
    geometric_ratio: float
    hybrid_ratio: float


class SpatialCorpus:
    """
    Synthetic corpus of spatial expressions for analysis.

    Based on patterns from SpaceEval (Pustejovsky et al., 2015) and
    cognitive studies (Coventry & Garrod, 2004).
    """

    # Preposition classification based on Landau (2024)
    PREPOSITION_CATEGORIES = {
        # Primarily geometric (projective)
        "above": PrepositionCategory.GEOMETRIC,
        "below": PrepositionCategory.GEOMETRIC,
        "over": PrepositionCategory.HYBRID,  # Can be functional (protection)
        "under": PrepositionCategory.HYBRID,
        "behind": PrepositionCategory.GEOMETRIC,
        "in front of": PrepositionCategory.GEOMETRIC,
        "left of": PrepositionCategory.GEOMETRIC,
        "right of": PrepositionCategory.GEOMETRIC,
        "beside": PrepositionCategory.GEOMETRIC,
        "between": PrepositionCategory.GEOMETRIC,

        # Primarily functional (force-dynamic)
        "on": PrepositionCategory.FUNCTIONAL,   # Support
        "in": PrepositionCategory.FUNCTIONAL,   # Containment
        "inside": PrepositionCategory.FUNCTIONAL,
        "within": PrepositionCategory.FUNCTIONAL,
        "atop": PrepositionCategory.FUNCTIONAL,
        "upon": PrepositionCategory.FUNCTIONAL,

        # Hybrid/context-dependent
        "at": PrepositionCategory.HYBRID,
        "by": PrepositionCategory.HYBRID,
        "near": PrepositionCategory.HYBRID,
        "next to": PrepositionCategory.HYBRID,
        "against": PrepositionCategory.HYBRID,
        "around": PrepositionCategory.HYBRID,
    }

    # Sentence templates with expected properties
    TEMPLATES = {
        "functional_support": [
            ("The {obj1} is on the {obj2}", {"prep": "on", "sense": "support"}),
            ("A {obj1} sits on the {obj2}", {"prep": "on", "sense": "support"}),
            ("The {obj1} rests on the {obj2}", {"prep": "on", "sense": "support"}),
            ("The {obj1} hangs on the {obj2}", {"prep": "on", "sense": "hanging"}),
            ("The {obj1} is stuck on the {obj2}", {"prep": "on", "sense": "adhesion"}),
        ],
        "functional_containment": [
            ("The {obj1} is in the {obj2}", {"prep": "in", "sense": "containment"}),
            ("A {obj1} floats in the {obj2}", {"prep": "in", "sense": "containment"}),
            ("The {obj1} is inside the {obj2}", {"prep": "inside", "sense": "containment"}),
            ("The {obj1} is within the {obj2}", {"prep": "within", "sense": "containment"}),
        ],
        "projective_geometric": [
            ("The {obj1} is above the {obj2}", {"prep": "above", "sense": "vertical"}),
            ("The {obj1} is below the {obj2}", {"prep": "below", "sense": "vertical"}),
            ("The {obj1} is behind the {obj2}", {"prep": "behind", "sense": "projective"}),
            ("The {obj1} is in front of the {obj2}", {"prep": "in front of", "sense": "projective"}),
            ("The {obj1} is left of the {obj2}", {"prep": "left of", "sense": "lateral"}),
            ("The {obj1} is beside the {obj2}", {"prep": "beside", "sense": "lateral"}),
        ],
        "proximity": [
            ("The {obj1} is near the {obj2}", {"prep": "near", "sense": "proximity"}),
            ("The {obj1} is at the {obj2}", {"prep": "at", "sense": "location"}),
            ("The {obj1} is by the {obj2}", {"prep": "by", "sense": "adjacency"}),
            ("The {obj1} is next to the {obj2}", {"prep": "next to", "sense": "adjacency"}),
        ],
        "complex": [
            ("The {obj1} in the {obj2} is on the {obj3}", {"prep": ["in", "on"], "sense": "chain"}),
            ("The {obj1} on the {obj2} is above the {obj3}", {"prep": ["on", "above"], "sense": "mixed"}),
        ],
    }

    # Object pairs with affordance information
    OBJECT_PAIRS = {
        "support": [
            ("cup", "table"),
            ("book", "shelf"),
            ("lamp", "desk"),
            ("plate", "counter"),
            ("phone", "nightstand"),
            ("vase", "mantle"),
            ("laptop", "bed"),
            ("keys", "hook"),
            ("picture", "wall"),
            ("poster", "door"),
        ],
        "containment": [
            ("flowers", "vase"),
            ("water", "cup"),
            ("coins", "jar"),
            ("clothes", "closet"),
            ("cat", "room"),
            ("fish", "tank"),
            ("letter", "envelope"),
            ("toys", "box"),
            ("fruit", "bowl"),
            ("people", "building"),
        ],
        "projective": [
            ("lamp", "table"),
            ("cloud", "mountain"),
            ("bird", "tree"),
            ("sun", "horizon"),
            ("plane", "city"),
            ("helicopter", "building"),
            ("moon", "lake"),
            ("kite", "field"),
        ],
    }

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.sentences = []
        self.triples = []

    def generate(self, n_per_category: int = 20) -> List[Dict[str, Any]]:
        """Generate a balanced corpus of spatial sentences."""
        corpus = []

        for category, templates in self.TEMPLATES.items():
            if category == "complex":
                continue  # Handle separately

            for _ in range(n_per_category):
                template, props = self.random.choice(templates)

                # Select appropriate object pair
                if "support" in category:
                    obj1, obj2 = self.random.choice(self.OBJECT_PAIRS["support"])
                elif "containment" in category:
                    obj1, obj2 = self.random.choice(self.OBJECT_PAIRS["containment"])
                else:
                    obj1, obj2 = self.random.choice(self.OBJECT_PAIRS["projective"])

                sentence = template.format(obj1=obj1, obj2=obj2)

                prep = props["prep"]
                prep_category = self.PREPOSITION_CATEGORIES.get(
                    prep, PrepositionCategory.HYBRID
                )

                entry = {
                    "sentence": sentence,
                    "category": category,
                    "preposition": prep,
                    "preposition_category": prep_category.value,
                    "sense": props["sense"],
                    "trajector": obj1,
                    "landmark": obj2,
                    "has_functional": prep_category == PrepositionCategory.FUNCTIONAL,
                }
                corpus.append(entry)

        self.sentences = corpus
        return corpus

    def get_statistics(self) -> CorpusStatistics:
        """Compute corpus statistics."""
        if not self.sentences:
            self.generate()

        prep_counts = defaultdict(int)
        sense_counts = defaultdict(int)
        cat_counts = defaultdict(int)

        for entry in self.sentences:
            prep_counts[entry["preposition"]] += 1
            sense_counts[entry["sense"]] += 1
            cat_counts[entry["preposition_category"]] += 1

        total = len(self.sentences)

        return CorpusStatistics(
            total_sentences=total,
            total_triples=total,  # 1:1 for simple sentences
            preposition_counts=dict(prep_counts),
            sense_distribution=dict(sense_counts),
            functional_ratio=cat_counts["functional"] / total if total else 0,
            geometric_ratio=cat_counts["geometric"] / total if total else 0,
            hybrid_ratio=cat_counts["hybrid"] / total if total else 0,
        )


class ComplexityExperiment:
    """
    Experiment comparing complexity across constraint fragment types.

    Tests the hypothesis that:
    - Pure geometric fragments are tractable (polynomial)
    - Pure functional fragments require specialized algorithms
    - Hybrid fragments approach NP-hardness
    """

    def __init__(self):
        self.analyzer = ComplexityAnalyzer()
        self.reasoner = FAQSRReasoner()
        self.results = []

    def create_geometric_network(self, n_vars: int, density: float = 0.5) -> FAQSRNetwork:
        """Create a network with only geometric constraints."""
        network = FAQSRNetwork()

        vars = [f"V{i}" for i in range(n_vars)]
        for v in vars:
            network.add_variable(v)

        # Add geometric constraints based on density
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if self.random.random() < density:
                    # Choose tractable RCC-8 relations
                    rel_choice = self.random.choice([
                        relation_set(RCC8Relation.DC),
                        relation_set(RCC8Relation.EC),
                        relation_set(RCC8Relation.TPP, RCC8Relation.NTPP),
                        relation_set(RCC8Relation.TPPi, RCC8Relation.NTPPi),
                        relation_set(RCC8Relation.DC, RCC8Relation.EC),
                    ])
                    network.add_geometric_constraint(vars[i], vars[j], rel_choice)

        return network

    def create_functional_network(self, n_vars: int, density: float = 0.5) -> FAQSRNetwork:
        """Create a network with functional constraints."""
        network = FAQSRNetwork()

        vars = [f"V{i}" for i in range(n_vars)]
        for v in vars:
            network.add_variable(v)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if self.random.random() < density:
                    func_choice = self.random.choice([
                        frozenset([FunctionalRelation.FSUPPORT]),
                        frozenset([FunctionalRelation.FCONTAIN]),
                        frozenset([FunctionalRelation.FCONTAIN_PARTIAL]),
                    ])
                    network.add_functional_constraint(vars[i], vars[j], func_choice)

        return network

    def create_hybrid_network(self, n_vars: int, density: float = 0.5) -> FAQSRNetwork:
        """Create a network mixing geometric and functional constraints."""
        network = FAQSRNetwork()

        vars = [f"V{i}" for i in range(n_vars)]
        for v in vars:
            network.add_variable(v)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if self.random.random() < density:
                    if self.random.random() < 0.5:
                        # Geometric
                        rel_choice = relation_set(RCC8Relation.EC, RCC8Relation.PO)
                        network.add_geometric_constraint(vars[i], vars[j], rel_choice)
                    else:
                        # Functional
                        func_choice = frozenset([FunctionalRelation.FSUPPORT])
                        network.add_functional_constraint(vars[i], vars[j], func_choice)

        return network

    def run_complexity_comparison(self, sizes: List[int] = None,
                                   trials: int = 10,
                                   seed: int = 42) -> ExperimentResult:
        """
        Compare complexity characteristics across fragment types.

        Returns detailed results on tractability, consistency checking time,
        and fragment classification.
        """
        if sizes is None:
            sizes = [3, 5, 7, 10, 15, 20]

        self.random = random.Random(seed)
        results_data = []

        for n_vars in sizes:
            for trial in range(trials):
                for fragment_type in ["geometric", "functional", "hybrid"]:
                    # Create network
                    if fragment_type == "geometric":
                        network = self.create_geometric_network(n_vars)
                    elif fragment_type == "functional":
                        network = self.create_functional_network(n_vars)
                    else:
                        network = self.create_hybrid_network(n_vars)

                    # Analyze complexity
                    complexity_result = self.analyzer.analyze(network)

                    # Time consistency checking
                    start_time = time.perf_counter()
                    consistency_result = self.reasoner.check_consistency(network)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    results_data.append({
                        "n_vars": n_vars,
                        "trial": trial,
                        "fragment_type": fragment_type,
                        "complexity_class": complexity_result.complexity_class.value,
                        "is_tractable": complexity_result.is_tractable,
                        "tractable_fraction": complexity_result.tractable_fraction,
                        "consistency_time_ms": elapsed_ms,
                        "is_consistent": consistency_result.status == ConsistencyStatus.CONSISTENT,
                        "cognitive_difficulty": complexity_result.predicted_processing_difficulty,
                    })

        # Aggregate metrics
        metrics = self._aggregate_complexity_metrics(results_data)

        return ExperimentResult(
            experiment_name="complexity_comparison",
            parameters={"sizes": sizes, "trials": trials},
            metrics=metrics,
            raw_data=results_data,
        )

    def _aggregate_complexity_metrics(self, data: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics from raw complexity data."""
        metrics = {}

        for ftype in ["geometric", "functional", "hybrid"]:
            ftype_data = [d for d in data if d["fragment_type"] == ftype]
            if not ftype_data:
                continue

            metrics[f"{ftype}_tractable_pct"] = (
                sum(1 for d in ftype_data if d["is_tractable"]) / len(ftype_data) * 100
            )
            metrics[f"{ftype}_avg_time_ms"] = (
                sum(d["consistency_time_ms"] for d in ftype_data) / len(ftype_data)
            )
            metrics[f"{ftype}_consistent_pct"] = (
                sum(1 for d in ftype_data if d["is_consistent"]) / len(ftype_data) * 100
            )

        return metrics


class ConsistencyExperiment:
    """
    Experiment evaluating consistency checking across network configurations.

    Tests:
    - Path consistency completeness for tractable fragments
    - Detection of geometric vs functional inconsistencies
    - Performance scaling with network size
    """

    def __init__(self):
        self.reasoner = FAQSRReasoner()
        self.random = None

    def create_consistent_chain(self, n_vars: int) -> FAQSRNetwork:
        """Create a consistent chain of containment relations."""
        network = FAQSRNetwork()

        vars = [f"V{i}" for i in range(n_vars)]
        for v in vars:
            network.add_variable(v)

        # Chain: V0 in V1, V1 in V2, ...
        for i in range(n_vars - 1):
            network.add_geometric_constraint(
                vars[i], vars[i+1],
                relation_set(RCC8Relation.TPP, RCC8Relation.NTPP)
            )

        return network

    def create_inconsistent_cycle(self, n_vars: int) -> FAQSRNetwork:
        """Create an inconsistent cyclic containment network."""
        network = FAQSRNetwork()

        vars = [f"V{i}" for i in range(n_vars)]
        for v in vars:
            network.add_variable(v)

        # Cycle: V0 in V1, V1 in V2, ..., V(n-1) in V0 (IMPOSSIBLE)
        for i in range(n_vars):
            next_i = (i + 1) % n_vars
            network.add_geometric_constraint(
                vars[i], vars[next_i],
                relation_set(RCC8Relation.NTPP)
            )

        return network

    def create_functional_inconsistency(self) -> FAQSRNetwork:
        """Create a network with functional-geometric conflict."""
        network = FAQSRNetwork()

        network.add_variable("A")
        network.add_variable("B")

        # A supports B (requires contact)
        network.add_functional_constraint(
            "A", "B",
            frozenset([FunctionalRelation.FSUPPORT])
        )

        # But also A is disconnected from B (contradicts support)
        network.add_geometric_constraint(
            "A", "B",
            relation_set(RCC8Relation.DC)
        )

        return network

    def run_consistency_benchmark(self, sizes: List[int] = None,
                                   trials: int = 20,
                                   seed: int = 42) -> ExperimentResult:
        """Run consistency checking benchmark."""
        if sizes is None:
            sizes = [3, 5, 10, 15, 20, 30, 50]

        self.random = random.Random(seed)
        results_data = []

        for n_vars in sizes:
            for trial in range(trials):
                # Test consistent chain
                network = self.create_consistent_chain(n_vars)

                start = time.perf_counter()
                result = self.reasoner.check_consistency(network)
                elapsed = (time.perf_counter() - start) * 1000

                results_data.append({
                    "n_vars": n_vars,
                    "trial": trial,
                    "network_type": "consistent_chain",
                    "is_consistent": result.status == ConsistencyStatus.CONSISTENT,
                    "time_ms": elapsed,
                    "expected_consistent": True,
                })

                # Test inconsistent cycle (only for small n to avoid too slow)
                if n_vars <= 20:
                    network = self.create_inconsistent_cycle(n_vars)

                    start = time.perf_counter()
                    result = self.reasoner.check_consistency(network)
                    elapsed = (time.perf_counter() - start) * 1000

                    results_data.append({
                        "n_vars": n_vars,
                        "trial": trial,
                        "network_type": "inconsistent_cycle",
                        "is_consistent": result.status == ConsistencyStatus.CONSISTENT,
                        "time_ms": elapsed,
                        "expected_consistent": False,
                    })

        # Compute accuracy metrics
        correct = sum(1 for d in results_data
                     if d["is_consistent"] == d["expected_consistent"])
        accuracy = correct / len(results_data) if results_data else 0

        avg_time = sum(d["time_ms"] for d in results_data) / len(results_data)

        return ExperimentResult(
            experiment_name="consistency_benchmark",
            parameters={"sizes": sizes, "trials": trials},
            metrics={
                "accuracy": accuracy * 100,
                "avg_time_ms": avg_time,
                "total_tests": len(results_data),
            },
            raw_data=results_data,
        )


class TranslationExperiment:
    """
    Experiment evaluating the SpRL -> GUM -> FA-QSR translation pipeline.

    Measures:
    - Preposition sense disambiguation accuracy
    - Constraint generation correctness
    - Coverage of spatial expressions
    """

    def __init__(self):
        self.translator = GUMToFAQSRTranslator()
        self.disambiguator = PrepositionDisambiguator()
        self.labeler = SpatialRoleLabeler(use_heuristics=True)

    def evaluate_sense_disambiguation(self, corpus: SpatialCorpus) -> ExperimentResult:
        """Evaluate preposition sense disambiguation."""
        if not corpus.sentences:
            corpus.generate()

        results_data = []
        correct = 0
        total = 0

        for entry in corpus.sentences:
            triple = SpatialTriple(
                trajector=entry["trajector"],
                indicator=entry["preposition"],
                landmark=entry["landmark"]
            )

            # Get predicted sense (returns SenseDisambiguation object)
            disambiguation = self.disambiguator.disambiguate(triple)
            predicted_sense = disambiguation.predicted_sense

            # Check if sense category matches expected
            expected_functional = entry["has_functional"]
            predicted_functional = disambiguation.is_functional

            is_correct = expected_functional == predicted_functional
            if is_correct:
                correct += 1
            total += 1

            results_data.append({
                "sentence": entry["sentence"],
                "preposition": entry["preposition"],
                "expected_functional": expected_functional,
                "predicted_sense": predicted_sense.value if predicted_sense else "unknown",
                "predicted_functional": predicted_functional,
                "correct": is_correct,
            })

        accuracy = correct / total if total else 0

        # Breakdown by preposition
        prep_accuracy = {}
        for prep in set(d["preposition"] for d in results_data):
            prep_data = [d for d in results_data if d["preposition"] == prep]
            prep_correct = sum(1 for d in prep_data if d["correct"])
            prep_accuracy[prep] = prep_correct / len(prep_data) if prep_data else 0

        return ExperimentResult(
            experiment_name="sense_disambiguation",
            parameters={},
            metrics={
                "overall_accuracy": accuracy * 100,
                "functional_precision": self._calc_precision(results_data, True),
                "geometric_precision": self._calc_precision(results_data, False),
                **{f"accuracy_{p}": v * 100 for p, v in prep_accuracy.items()},
            },
            raw_data=results_data,
        )

    def _calc_precision(self, data: List[Dict], for_functional: bool) -> float:
        """Calculate precision for functional or geometric classification."""
        predicted = [d for d in data if d["predicted_functional"] == for_functional]
        if not predicted:
            return 0.0
        correct = sum(1 for d in predicted if d["correct"])
        return correct / len(predicted) * 100

    def evaluate_constraint_generation(self, corpus: SpatialCorpus) -> ExperimentResult:
        """Evaluate constraint generation from triples."""
        if not corpus.sentences:
            corpus.generate()

        results_data = []

        for entry in corpus.sentences:
            triple = SpatialTriple(
                trajector=entry["trajector"],
                indicator=entry["preposition"],
                landmark=entry["landmark"]
            )

            # Translate to FA-QSR constraint
            translation = self.translator.translate_triple(triple)

            has_functional = len(translation.constraint.functional) > 0
            has_geometric = translation.constraint.geometric != UNIVERSAL

            # Check if constraint type matches expectation
            expected_functional = entry["has_functional"]

            results_data.append({
                "sentence": entry["sentence"],
                "preposition": entry["preposition"],
                "expected_functional": expected_functional,
                "generated_functional": has_functional,
                "generated_geometric": has_geometric,
                "complexity": translation.complexity_class,
                "confidence": translation.confidence,
                "correct_type": expected_functional == has_functional,
            })

        accuracy = sum(1 for d in results_data if d["correct_type"]) / len(results_data)
        avg_confidence = sum(d["confidence"] for d in results_data) / len(results_data)

        return ExperimentResult(
            experiment_name="constraint_generation",
            parameters={},
            metrics={
                "type_accuracy": accuracy * 100,
                "avg_confidence": avg_confidence,
                "functional_coverage": sum(1 for d in results_data if d["generated_functional"]) / len(results_data) * 100,
            },
            raw_data=results_data,
        )


class CognitiveDifficultyExperiment:
    """
    Experiment correlating computational complexity with cognitive difficulty.

    Based on Landau (2024): geometric terms are processed differently than
    functional terms, with functional terms requiring more world knowledge.
    """

    def __init__(self):
        self.analyzer = ComplexityAnalyzer()

    def run_cognitive_correlation(self, corpus: SpatialCorpus) -> ExperimentResult:
        """
        Analyze correlation between fragment type and predicted cognitive load.
        """
        if not corpus.sentences:
            corpus.generate()

        from network_translator.gum_translator import GUMToFAQSRTranslator
        from network_translator.sprl_to_network import NetworkBuilder

        translator = GUMToFAQSRTranslator()
        builder = NetworkBuilder()

        results_data = []

        # Group sentences by category
        by_category = defaultdict(list)
        for entry in corpus.sentences:
            by_category[entry["category"]].append(entry)

        for category, entries in by_category.items():
            for entry in entries:
                # Create single-triple network
                triple = SpatialTriple(
                    trajector=entry["trajector"],
                    indicator=entry["preposition"],
                    landmark=entry["landmark"]
                )

                build_result = builder.build_from_triples([triple])
                complexity = self.analyzer.analyze(build_result.network)

                results_data.append({
                    "category": category,
                    "preposition": entry["preposition"],
                    "preposition_category": entry["preposition_category"],
                    "fragment_type": complexity.fragment_type.value,
                    "complexity_class": complexity.complexity_class.value,
                    "is_tractable": complexity.is_tractable,
                    "predicted_difficulty": complexity.predicted_processing_difficulty,
                })

        # Compute correlations
        difficulty_map = {"easy": 1, "medium": 2, "hard": 3, "unknown": 2}

        by_prep_cat = defaultdict(list)
        for d in results_data:
            by_prep_cat[d["preposition_category"]].append(
                difficulty_map.get(d["predicted_difficulty"], 2)
            )

        avg_difficulty = {
            cat: sum(vals) / len(vals) if vals else 0
            for cat, vals in by_prep_cat.items()
        }

        # Hypothesis: functional should have higher difficulty
        functional_diff = avg_difficulty.get("functional", 0)
        geometric_diff = avg_difficulty.get("geometric", 0)

        return ExperimentResult(
            experiment_name="cognitive_correlation",
            parameters={},
            metrics={
                "functional_avg_difficulty": functional_diff,
                "geometric_avg_difficulty": geometric_diff,
                "difficulty_difference": functional_diff - geometric_diff,
                "supports_hypothesis": functional_diff > geometric_diff,
            },
            raw_data=results_data,
            notes=[
                "Hypothesis: Functional prepositions require more cognitive effort",
                f"Functional mean difficulty: {functional_diff:.2f}",
                f"Geometric mean difficulty: {geometric_diff:.2f}",
            ]
        )


class EmpiricalAnalyzer:
    """
    Main class orchestrating all empirical analyses.

    Produces comprehensive results for the paper's Analysis section.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.corpus = SpatialCorpus(seed=seed)
        self.results = {}

    def run_all_experiments(self, verbose: bool = True) -> Dict[str, ExperimentResult]:
        """Run all experiments and collect results."""

        if verbose:
            print("=" * 70)
            print("FA-QSR Empirical Analysis")
            print("=" * 70)

        # Generate corpus
        if verbose:
            print("\n1. Generating spatial corpus...")
        self.corpus.generate(n_per_category=25)
        stats = self.corpus.get_statistics()
        if verbose:
            print(f"   Generated {stats.total_sentences} sentences")
            print(f"   Functional ratio: {stats.functional_ratio:.1%}")
            print(f"   Geometric ratio: {stats.geometric_ratio:.1%}")

        # Complexity experiment
        if verbose:
            print("\n2. Running complexity comparison experiment...")
        complexity_exp = ComplexityExperiment()
        self.results["complexity"] = complexity_exp.run_complexity_comparison(
            sizes=[3, 5, 7, 10, 15],
            trials=5,
            seed=self.seed
        )
        if verbose:
            m = self.results["complexity"].metrics
            print(f"   Geometric tractable: {m.get('geometric_tractable_pct', 0):.1f}%")
            print(f"   Hybrid tractable: {m.get('hybrid_tractable_pct', 0):.1f}%")

        # Consistency experiment
        if verbose:
            print("\n3. Running consistency benchmark...")
        consistency_exp = ConsistencyExperiment()
        self.results["consistency"] = consistency_exp.run_consistency_benchmark(
            sizes=[3, 5, 10, 15, 20],
            trials=10,
            seed=self.seed
        )
        if verbose:
            m = self.results["consistency"].metrics
            print(f"   Accuracy: {m.get('accuracy', 0):.1f}%")
            print(f"   Avg time: {m.get('avg_time_ms', 0):.2f}ms")

        # Translation experiment
        if verbose:
            print("\n4. Running translation pipeline evaluation...")
        translation_exp = TranslationExperiment()
        self.results["sense_disambiguation"] = translation_exp.evaluate_sense_disambiguation(
            self.corpus
        )
        self.results["constraint_generation"] = translation_exp.evaluate_constraint_generation(
            self.corpus
        )
        if verbose:
            m = self.results["sense_disambiguation"].metrics
            print(f"   Sense disambiguation accuracy: {m.get('overall_accuracy', 0):.1f}%")
            m = self.results["constraint_generation"].metrics
            print(f"   Constraint type accuracy: {m.get('type_accuracy', 0):.1f}%")

        # Cognitive difficulty experiment
        if verbose:
            print("\n5. Running cognitive difficulty analysis...")
        cognitive_exp = CognitiveDifficultyExperiment()
        self.results["cognitive"] = cognitive_exp.run_cognitive_correlation(self.corpus)
        if verbose:
            m = self.results["cognitive"].metrics
            print(f"   Functional difficulty: {m.get('functional_avg_difficulty', 0):.2f}")
            print(f"   Geometric difficulty: {m.get('geometric_avg_difficulty', 0):.2f}")
            print(f"   Supports hypothesis: {m.get('supports_hypothesis', False)}")

        if verbose:
            print("\n" + "=" * 70)
            print("Analysis Complete")
            print("=" * 70)

        return self.results

    def generate_latex_tables(self) -> str:
        """Generate LaTeX tables from results."""
        tables = []

        # Table 1: Complexity Comparison
        if "complexity" in self.results:
            m = self.results["complexity"].metrics
            tables.append("""
\\begin{{table}}[h]
\\centering
\\caption{{Complexity Analysis by Fragment Type}}
\\label{{tab:complexity}}
\\begin{{tabular}}{{lccc}}
\\toprule
Fragment Type & Tractable (\\%) & Avg. Time (ms) & Consistent (\\%) \\\\
\\midrule
Geometric & {:.1f} & {:.2f} & {:.1f} \\\\
Functional & {:.1f} & {:.2f} & {:.1f} \\\\
Hybrid & {:.1f} & {:.2f} & {:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".format(
                m.get("geometric_tractable_pct", 0),
                m.get("geometric_avg_time_ms", 0),
                m.get("geometric_consistent_pct", 0),
                m.get("functional_tractable_pct", 0),
                m.get("functional_avg_time_ms", 0),
                m.get("functional_consistent_pct", 0),
                m.get("hybrid_tractable_pct", 0),
                m.get("hybrid_avg_time_ms", 0),
                m.get("hybrid_consistent_pct", 0),
            ))

        # Table 2: Translation Accuracy
        if "sense_disambiguation" in self.results:
            m = self.results["sense_disambiguation"].metrics
            tables.append("""
\\begin{{table}}[h]
\\centering
\\caption{{Preposition Sense Disambiguation Results}}
\\label{{tab:disambiguation}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value (\\%) \\\\
\\midrule
Overall Accuracy & {:.1f} \\\\
Functional Precision & {:.1f} \\\\
Geometric Precision & {:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".format(
                m.get("overall_accuracy", 0),
                m.get("functional_precision", 0),
                m.get("geometric_precision", 0),
            ))

        # Table 3: Cognitive Correlation
        if "cognitive" in self.results:
            m = self.results["cognitive"].metrics
            tables.append("""
\\begin{{table}}[h]
\\centering
\\caption{{Predicted Cognitive Difficulty by Preposition Category}}
\\label{{tab:cognitive}}
\\begin{{tabular}}{{lc}}
\\toprule
Preposition Category & Mean Difficulty (1-3) \\\\
\\midrule
Geometric (above, below, etc.) & {:.2f} \\\\
Functional (on, in) & {:.2f} \\\\
Difference & {:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".format(
                m.get("geometric_avg_difficulty", 0),
                m.get("functional_avg_difficulty", 0),
                m.get("difficulty_difference", 0),
            ))

        return "\n".join(tables)

    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the paper."""
        summary = {
            "corpus": self.corpus.get_statistics().__dict__,
        }

        for name, result in self.results.items():
            summary[name] = result.metrics

        return summary

    def save_results(self, filepath: str):
        """Save all results to JSON."""
        output = {
            "corpus_statistics": self.corpus.get_statistics().__dict__,
            "experiments": {}
        }

        for name, result in self.results.items():
            output["experiments"][name] = {
                "name": result.experiment_name,
                "parameters": result.parameters,
                "metrics": result.metrics,
                "notes": result.notes,
            }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)


def main():
    """Run full empirical analysis."""
    analyzer = EmpiricalAnalyzer(seed=42)
    results = analyzer.run_all_experiments(verbose=True)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS FOR PAPER")
    print("=" * 70)

    summary = analyzer.generate_summary_statistics()

    print("\nCorpus Statistics:")
    print(f"  Total sentences: {summary['corpus']['total_sentences']}")
    print(f"  Functional ratio: {summary['corpus']['functional_ratio']:.1%}")
    print(f"  Geometric ratio: {summary['corpus']['geometric_ratio']:.1%}")

    print("\nKey Findings:")
    if "complexity" in summary:
        m = summary["complexity"]
        print(f"  1. Pure geometric fragments: {m.get('geometric_tractable_pct', 0):.0f}% tractable")
        print(f"  2. Hybrid fragments: {m.get('hybrid_tractable_pct', 0):.0f}% tractable")
        print(f"  3. Geometric avg. consistency time: {m.get('geometric_avg_time_ms', 0):.2f}ms")
        print(f"  4. Hybrid avg. consistency time: {m.get('hybrid_avg_time_ms', 0):.2f}ms")

    if "consistency" in summary:
        m = summary["consistency"]
        print(f"  5. Path consistency accuracy: {m.get('accuracy', 0):.0f}%")

    if "sense_disambiguation" in summary:
        m = summary["sense_disambiguation"]
        print(f"  6. Sense disambiguation accuracy: {m.get('overall_accuracy', 0):.0f}%")

    if "cognitive" in summary:
        m = summary["cognitive"]
        supports = m.get('supports_hypothesis', False)
        print(f"  7. Cognitive hypothesis supported: {supports}")

    # Generate LaTeX tables
    print("\n" + "=" * 70)
    print("LATEX TABLES")
    print("=" * 70)
    print(analyzer.generate_latex_tables())

    # Save results
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "empirical_results.json"
    )
    analyzer.save_results(results_path)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
