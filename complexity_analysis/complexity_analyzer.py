"""
Complexity Analyzer for FA-QSR

This module analyzes the computational complexity of FA-QSR constraint
networks, identifying tractable vs intractable fragments based on the
theoretical analysis of Renz & Nebel (1999) and extensions for functional
constraints.

Key results from the analysis:
1. Pure RCC-8 Horn fragments (H8) are polynomial-time solvable
2. Functional constraints alone may be tractable (specialized algorithms)
3. Mixed geometric-functional constraints approach NP-hardness
4. Complexity correlates with cognitive processing difficulty (Landau, 2024)

References:
- Renz & Nebel (1999) - On the Complexity of Qualitative Spatial Reasoning
- Cohn & Renz (2008) - QSR Complexity Results
- Landau (2024) - Geometric vs Functional Terms in Cognition
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsr_base.rcc8 import (
    RCC8Relation,
    RelationSet,
    UNIVERSAL,
    TractableFragments,
)
from qsr_base.faqsr_calculus import (
    FAQSRNetwork,
    FAQSRConstraint,
    FunctionalRelation,
)


class ComplexityClass(Enum):
    """Computational complexity classes for constraint satisfaction."""
    POLYNOMIAL = "polynomial"       # O(n^k) for some k
    NP_COMPLETE = "np-complete"     # NP-complete
    NP_HARD = "np-hard"            # At least as hard as NP-complete
    PSPACE = "pspace"              # Polynomial space
    UNDECIDABLE = "undecidable"    # Not decidable
    UNKNOWN = "unknown"            # Cannot determine


class FragmentType(Enum):
    """Types of constraint fragments."""
    PURE_GEOMETRIC = "pure_geometric"
    PURE_FUNCTIONAL = "pure_functional"
    HYBRID = "hybrid"
    PROJECTIVE = "projective"
    EMPTY = "empty"


@dataclass
class ComplexityResult:
    """
    Result of complexity analysis for a constraint network.
    """
    # Overall complexity class
    complexity_class: ComplexityClass

    # Fragment analysis
    fragment_type: FragmentType
    num_geometric_constraints: int
    num_functional_constraints: int
    num_hybrid_constraints: int

    # Tractability details
    is_tractable: bool
    tractable_fraction: float  # Fraction in tractable fragment

    # Specific findings
    problematic_constraints: List[str] = field(default_factory=list)
    tractability_violations: List[str] = field(default_factory=list)

    # Cognitive prediction
    predicted_processing_difficulty: str = "unknown"  # easy, medium, hard

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FragmentAnalysis:
    """Analysis of a specific constraint fragment."""
    constraints: List[FAQSRConstraint]
    fragment_type: FragmentType
    is_tractable: bool
    complexity_class: ComplexityClass
    notes: List[str] = field(default_factory=list)


class TractabilityChecker:
    """
    Checks tractability of RCC-8 constraint sets.

    Based on the maximal tractable subsets identified by
    Renz & Nebel (1999): H8, C8, Q8.
    """

    # The 8 base relations
    BASE_RELATIONS = frozenset(RCC8Relation)

    # Known tractable constraints (simplified H8 membership)
    # Full H8 has 148 relations - this is a representative subset
    H8_TRACTABLE = {
        # Singletons
        frozenset([RCC8Relation.DC]),
        frozenset([RCC8Relation.EC]),
        frozenset([RCC8Relation.PO]),
        frozenset([RCC8Relation.EQ]),
        frozenset([RCC8Relation.TPP]),
        frozenset([RCC8Relation.NTPP]),
        frozenset([RCC8Relation.TPPi]),
        frozenset([RCC8Relation.NTPPi]),

        # Pairs preserving tractability
        frozenset([RCC8Relation.DC, RCC8Relation.EC]),
        frozenset([RCC8Relation.TPP, RCC8Relation.NTPP]),
        frozenset([RCC8Relation.TPPi, RCC8Relation.NTPPi]),
        frozenset([RCC8Relation.TPP, RCC8Relation.EQ]),
        frozenset([RCC8Relation.TPPi, RCC8Relation.EQ]),
        frozenset([RCC8Relation.EC, RCC8Relation.PO]),

        # Triples
        frozenset([RCC8Relation.TPP, RCC8Relation.TPPi, RCC8Relation.EQ]),
        frozenset([RCC8Relation.DC, RCC8Relation.EC, RCC8Relation.PO]),

        # Larger tractable sets
        frozenset([RCC8Relation.DC, RCC8Relation.EC, RCC8Relation.PO,
                   RCC8Relation.TPP, RCC8Relation.NTPP]),
        frozenset([RCC8Relation.DC, RCC8Relation.EC, RCC8Relation.PO,
                   RCC8Relation.TPPi, RCC8Relation.NTPPi]),

        # Universal
        UNIVERSAL,
    }

    def is_in_H8(self, relations: RelationSet) -> bool:
        """
        Check if relation set is in the H8 tractable fragment.

        This is a simplified check. Full H8 membership requires
        checking against all 148 relations in the fragment.
        """
        # Empty is trivially in H8
        if not relations:
            return True

        # Check explicit membership
        if relations in self.H8_TRACTABLE:
            return True

        # Singletons are always in H8
        if len(relations) == 1:
            return True

        # Universal is in H8
        if relations == UNIVERSAL:
            return True

        # For other cases, use heuristic
        # Relations dominated by "part-of" structure tend to be tractable
        part_relations = {RCC8Relation.TPP, RCC8Relation.NTPP,
                         RCC8Relation.TPPi, RCC8Relation.NTPPi, RCC8Relation.EQ}
        if relations.issubset(part_relations):
            return True

        return False

    def check_network_tractability(self, constraints: List[RelationSet]) -> Tuple[bool, float]:
        """
        Check tractability of a network given its constraint sets.

        Returns:
            (is_tractable, fraction_tractable)
        """
        if not constraints:
            return True, 1.0

        tractable_count = sum(1 for c in constraints if self.is_in_H8(c))
        fraction = tractable_count / len(constraints)

        # Network is tractable if ALL constraints are in H8
        is_tractable = tractable_count == len(constraints)

        return is_tractable, fraction


class FragmentClassifier:
    """
    Classifies constraint fragments by their structure and complexity.
    """

    def classify(self, constraint: FAQSRConstraint) -> FragmentType:
        """Classify a single constraint by fragment type."""
        has_geometric = constraint.geometric != UNIVERSAL and len(constraint.geometric) > 0
        has_functional = len(constraint.functional) > 0

        if not has_geometric and not has_functional:
            return FragmentType.EMPTY
        elif has_geometric and has_functional:
            return FragmentType.HYBRID
        elif has_geometric:
            # Check if projective
            if self._is_projective(constraint.geometric):
                return FragmentType.PROJECTIVE
            return FragmentType.PURE_GEOMETRIC
        else:
            return FragmentType.PURE_FUNCTIONAL

    def _is_projective(self, relations: RelationSet) -> bool:
        """
        Check if relations correspond to projective constraints.

        Projective constraints typically involve DC/EC without contact.
        """
        # Simplified check - projective if only DC/EC and no parthood
        parthood = {RCC8Relation.TPP, RCC8Relation.NTPP,
                   RCC8Relation.TPPi, RCC8Relation.NTPPi}
        return not relations.intersection(parthood) and len(relations) <= 2

    def classify_network(self, network: FAQSRNetwork) -> Dict[FragmentType, int]:
        """Classify all constraints in a network by fragment type."""
        counts = {ft: 0 for ft in FragmentType}

        for constraint in network.get_all_constraints():
            ftype = self.classify(constraint)
            counts[ftype] += 1

        return counts


class ComplexityAnalyzer:
    """
    Main complexity analyzer for FA-QSR networks.

    Provides comprehensive analysis of computational complexity
    including tractability checking, fragment classification,
    and cognitive difficulty prediction.
    """

    def __init__(self):
        self.tractability_checker = TractabilityChecker()
        self.fragment_classifier = FragmentClassifier()

    def analyze(self, network: FAQSRNetwork) -> ComplexityResult:
        """
        Perform comprehensive complexity analysis on a network.

        Args:
            network: The FA-QSR network to analyze

        Returns:
            ComplexityResult with detailed analysis
        """
        constraints = network.get_all_constraints()

        if not constraints:
            return ComplexityResult(
                complexity_class=ComplexityClass.POLYNOMIAL,
                fragment_type=FragmentType.EMPTY,
                num_geometric_constraints=0,
                num_functional_constraints=0,
                num_hybrid_constraints=0,
                is_tractable=True,
                tractable_fraction=1.0,
                predicted_processing_difficulty="easy"
            )

        # Classify fragments
        fragment_counts = self.fragment_classifier.classify_network(network)

        # Determine dominant fragment type
        if fragment_counts[FragmentType.HYBRID] > 0:
            dominant_type = FragmentType.HYBRID
        elif fragment_counts[FragmentType.PURE_FUNCTIONAL] > 0:
            dominant_type = FragmentType.PURE_FUNCTIONAL
        elif fragment_counts[FragmentType.PROJECTIVE] > 0:
            dominant_type = FragmentType.PROJECTIVE
        elif fragment_counts[FragmentType.PURE_GEOMETRIC] > 0:
            dominant_type = FragmentType.PURE_GEOMETRIC
        else:
            dominant_type = FragmentType.EMPTY

        # Check tractability of geometric component
        geometric_constraints = [c.geometric for c in constraints]
        is_tractable, tractable_fraction = self.tractability_checker.check_network_tractability(
            geometric_constraints
        )

        # Identify problematic constraints
        problematic = []
        violations = []

        for constraint in constraints:
            # Check for non-tractable geometric
            if not self.tractability_checker.is_in_H8(constraint.geometric):
                violations.append(f"Non-H8 geometric: {constraint}")

            # Hybrid constraints are potentially problematic
            if self.fragment_classifier.classify(constraint) == FragmentType.HYBRID:
                problematic.append(f"Hybrid constraint: {constraint}")

        # Determine overall complexity class
        complexity_class = self._determine_complexity_class(
            dominant_type, is_tractable, fragment_counts
        )

        # Predict cognitive processing difficulty
        difficulty = self._predict_cognitive_difficulty(
            complexity_class, dominant_type, tractable_fraction
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            complexity_class, problematic, violations
        )

        return ComplexityResult(
            complexity_class=complexity_class,
            fragment_type=dominant_type,
            num_geometric_constraints=fragment_counts[FragmentType.PURE_GEOMETRIC] + fragment_counts[FragmentType.PROJECTIVE],
            num_functional_constraints=fragment_counts[FragmentType.PURE_FUNCTIONAL],
            num_hybrid_constraints=fragment_counts[FragmentType.HYBRID],
            is_tractable=is_tractable and fragment_counts[FragmentType.HYBRID] == 0,
            tractable_fraction=tractable_fraction,
            problematic_constraints=problematic,
            tractability_violations=violations,
            predicted_processing_difficulty=difficulty,
            recommendations=recommendations
        )

    def _determine_complexity_class(self, dominant_type: FragmentType,
                                   is_geometric_tractable: bool,
                                   counts: Dict[FragmentType, int]) -> ComplexityClass:
        """Determine overall complexity class."""
        # Hybrid constraints push toward NP-hardness
        if counts[FragmentType.HYBRID] > 0:
            return ComplexityClass.NP_HARD

        # Pure functional with specialized algorithms
        if dominant_type == FragmentType.PURE_FUNCTIONAL:
            # Functional reasoning may have specialized polynomial algorithms
            return ComplexityClass.POLYNOMIAL

        # Pure geometric - depends on fragment
        if dominant_type in (FragmentType.PURE_GEOMETRIC, FragmentType.PROJECTIVE):
            if is_geometric_tractable:
                return ComplexityClass.POLYNOMIAL
            else:
                return ComplexityClass.NP_COMPLETE

        return ComplexityClass.UNKNOWN

    def _predict_cognitive_difficulty(self, complexity: ComplexityClass,
                                     fragment_type: FragmentType,
                                     tractable_fraction: float) -> str:
        """
        Predict cognitive processing difficulty.

        Based on Landau (2024) findings that geometric terms align with
        tractable reasoning while functional terms require more cognitive effort.
        """
        if fragment_type == FragmentType.PURE_GEOMETRIC:
            if tractable_fraction >= 0.9:
                return "easy"
            elif tractable_fraction >= 0.5:
                return "medium"
            else:
                return "hard"

        elif fragment_type == FragmentType.PROJECTIVE:
            # Projective reasoning is well-studied and relatively easy
            return "easy"

        elif fragment_type == FragmentType.PURE_FUNCTIONAL:
            # Functional reasoning requires world knowledge
            return "medium"

        elif fragment_type == FragmentType.HYBRID:
            # Mixed reasoning is cognitively demanding
            return "hard"

        return "unknown"

    def _generate_recommendations(self, complexity: ComplexityClass,
                                 problematic: List[str],
                                 violations: List[str]) -> List[str]:
        """Generate recommendations for handling the network."""
        recommendations = []

        if complexity == ComplexityClass.NP_HARD:
            recommendations.append(
                "Consider decomposing hybrid constraints into separate "
                "geometric and functional sub-problems"
            )

        if violations:
            recommendations.append(
                f"Found {len(violations)} non-tractable geometric constraints. "
                "Consider approximating with tractable H8 relations"
            )

        if problematic:
            recommendations.append(
                f"Found {len(problematic)} potentially problematic constraints. "
                "These may benefit from incremental reasoning"
            )

        if complexity == ComplexityClass.POLYNOMIAL:
            recommendations.append(
                "Network is in tractable fragment. Path consistency is complete"
            )

        return recommendations

    def compare_fragments(self, network: FAQSRNetwork) -> Dict[str, Any]:
        """
        Compare complexity across different constraint fragments.

        Useful for validating the geometric/functional split hypothesis.
        """
        constraints = network.get_all_constraints()

        geometric_only = []
        functional_only = []
        hybrid = []

        for c in constraints:
            ftype = self.fragment_classifier.classify(c)
            if ftype == FragmentType.PURE_GEOMETRIC or ftype == FragmentType.PROJECTIVE:
                geometric_only.append(c.geometric)
            elif ftype == FragmentType.PURE_FUNCTIONAL:
                functional_only.append(c)
            elif ftype == FragmentType.HYBRID:
                hybrid.append(c)

        # Analyze each fragment
        geo_tractable, geo_fraction = self.tractability_checker.check_network_tractability(
            geometric_only
        )

        return {
            'geometric_constraints': len(geometric_only),
            'functional_constraints': len(functional_only),
            'hybrid_constraints': len(hybrid),
            'geometric_tractable': geo_tractable,
            'geometric_tractable_fraction': geo_fraction,
            'functional_complexity': 'specialized' if functional_only else 'none',
            'hybrid_complexity': 'np-hard' if hybrid else 'none',
            'overall_tractable': geo_tractable and len(hybrid) == 0,
        }


def demonstrate_complexity_analysis():
    """Demonstrate complexity analysis capabilities."""
    print("FA-QSR Complexity Analysis Demonstration")
    print("=" * 50)

    from qsr_base.rcc8 import relation_set

    analyzer = ComplexityAnalyzer()

    # Example 1: Pure geometric network (tractable)
    print("\n1. Pure Geometric Network (Tractable)")
    print("-" * 40)

    network1 = FAQSRNetwork()
    network1.add_variable("A")
    network1.add_variable("B")
    network1.add_variable("C")

    network1.add_geometric_constraint("A", "B",
                                     relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))
    network1.add_geometric_constraint("B", "C",
                                     relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))

    result1 = analyzer.analyze(network1)
    print(f"Complexity class: {result1.complexity_class.value}")
    print(f"Fragment type: {result1.fragment_type.value}")
    print(f"Is tractable: {result1.is_tractable}")
    print(f"Cognitive difficulty: {result1.predicted_processing_difficulty}")

    # Example 2: Hybrid network (potentially NP-hard)
    print("\n2. Hybrid Network (NP-Hard)")
    print("-" * 40)

    network2 = FAQSRNetwork()
    network2.add_variable("cup")
    network2.add_variable("table")
    network2.add_variable("vase")

    # Add functional constraint
    network2.add_functional_constraint("cup", "table",
                                       frozenset([FunctionalRelation.FSUPPORT]))
    network2.add_geometric_constraint("vase", "table",
                                     relation_set(RCC8Relation.EC, RCC8Relation.PO))

    result2 = analyzer.analyze(network2)
    print(f"Complexity class: {result2.complexity_class.value}")
    print(f"Fragment type: {result2.fragment_type.value}")
    print(f"Is tractable: {result2.is_tractable}")
    print(f"Hybrid constraints: {result2.num_hybrid_constraints}")
    print(f"Cognitive difficulty: {result2.predicted_processing_difficulty}")

    if result2.recommendations:
        print("Recommendations:")
        for rec in result2.recommendations:
            print(f"  - {rec}")

    # Example 3: Fragment comparison
    print("\n3. Fragment Comparison")
    print("-" * 40)

    comparison = analyzer.compare_fragments(network2)
    for key, value in comparison.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demonstrate_complexity_analysis()
