"""
Path Consistency (Algebraic Closure) Algorithm for FA-QSR

This module implements path consistency checking and constraint propagation
for both pure RCC-8 and hybrid FA-QSR networks.

Path consistency is a local consistency technique that ensures:
For all triples (i, j, k): R(i,j) ⊆ R(i,k) ∘ R(k,j)

For RCC-8, path consistency is complete for tractable fragments (H8),
meaning if a network is path consistent, it is satisfiable.

References:
- Renz & Nebel (1999) - On the Complexity of Qualitative Spatial Reasoning
- Cohn & Renz (2008) - Qualitative Spatial Representation and Reasoning
"""

from typing import Set, Tuple, Optional, List, Dict, Iterator
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsr_base.rcc8 import (
    RCC8Relation,
    RelationSet,
    UNIVERSAL,
    EMPTY,
    COMPOSITION_TABLE,
    ConstraintNetwork,
)
from qsr_base.faqsr_calculus import (
    FAQSRNetwork,
    FAQSRConstraint,
    FunctionalRelation,
    FunctionalRelationSet,
    FUNCTIONAL_COMPOSITION,
    GeometricRequirements,
)


class ConsistencyStatus(Enum):
    """Result status of consistency checking."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"  # Could not determine (timeout, etc.)


@dataclass
class ConsistencyResult:
    """
    Result of a consistency check.

    Attributes:
        status: Whether the network is consistent
        refined_network: The network after constraint propagation (if consistent)
        conflict: Description of inconsistency found (if inconsistent)
        iterations: Number of propagation iterations performed
        variables_involved: Variables involved in conflict (if any)
    """
    status: ConsistencyStatus
    refined_network: Optional[ConstraintNetwork] = None
    conflict: Optional[str] = None
    iterations: int = 0
    variables_involved: Optional[Tuple[str, ...]] = None

    def is_consistent(self) -> bool:
        return self.status == ConsistencyStatus.CONSISTENT


class PathConsistencyChecker:
    """
    Path consistency algorithm for RCC-8 constraint networks.

    Implements the PC-2 (or AC-3 style) algorithm for achieving
    path consistency through iterative constraint refinement.
    """

    def __init__(self, max_iterations: int = 10000):
        """
        Initialize the checker.

        Args:
            max_iterations: Maximum iterations before giving up
        """
        self.max_iterations = max_iterations
        self._composition = COMPOSITION_TABLE

    def check(self, network: ConstraintNetwork) -> ConsistencyResult:
        """
        Check path consistency of a constraint network.

        This algorithm iteratively refines constraints until either:
        1. A fixed point is reached (path consistent)
        2. An empty constraint is derived (inconsistent)
        3. Maximum iterations exceeded

        Args:
            network: The constraint network to check

        Returns:
            ConsistencyResult with status and refined network
        """
        # Work on a copy to not modify the original
        working = network.copy()
        variables = list(working.variables)
        n = len(variables)

        if n < 2:
            return ConsistencyResult(
                status=ConsistencyStatus.CONSISTENT,
                refined_network=working,
                iterations=0
            )

        # Initialize queue with all constraint pairs
        queue: Set[Tuple[str, str]] = set()
        for i in range(n):
            for j in range(i + 1, n):
                queue.add((variables[i], variables[j]))

        iterations = 0

        while queue and iterations < self.max_iterations:
            iterations += 1

            # Pop a pair to process
            vi, vj = queue.pop()

            # Try to refine using all intermediate variables
            for vk in variables:
                if vk == vi or vk == vj:
                    continue

                # Get current constraints
                rij = working.get_constraint(vi, vj)
                rik = working.get_constraint(vi, vk)
                rkj = working.get_constraint(vk, vj)

                # Compute composition rik ; rkj
                composed = self._composition.compose_sets(rik, rkj)

                # Refine rij
                refined = rij & composed

                if not refined:
                    # Inconsistency detected
                    return ConsistencyResult(
                        status=ConsistencyStatus.INCONSISTENT,
                        conflict=f"Empty constraint derived for ({vi}, {vj}) via {vk}",
                        iterations=iterations,
                        variables_involved=(vi, vj, vk)
                    )

                if refined != rij:
                    # Constraint was refined, update network
                    working._constraints[(min(vi, vj), max(vi, vj))] = (
                        refined if vi < vj else
                        frozenset(r.inverse() for r in refined)
                    )

                    # Add affected pairs back to queue
                    for vm in variables:
                        if vm != vi and vm != vj:
                            queue.add((min(vi, vm), max(vi, vm)))
                            queue.add((min(vj, vm), max(vj, vm)))

        if iterations >= self.max_iterations:
            return ConsistencyResult(
                status=ConsistencyStatus.UNKNOWN,
                conflict="Maximum iterations exceeded",
                iterations=iterations
            )

        return ConsistencyResult(
            status=ConsistencyStatus.CONSISTENT,
            refined_network=working,
            iterations=iterations
        )

    def enforce_path_consistency(self, network: ConstraintNetwork) -> Optional[ConstraintNetwork]:
        """
        Enforce path consistency on a network.

        Returns the refined network if consistent, None otherwise.
        """
        result = self.check(network)
        if result.is_consistent():
            return result.refined_network
        return None


class FAQSRReasoner:
    """
    Reasoning engine for FA-QSR networks.

    Extends path consistency to handle hybrid geometric-functional constraints.
    Uses a two-phase approach:
    1. Enforce functional requirements on geometric constraints
    2. Apply path consistency to the geometric component
    3. Check functional constraint consistency
    """

    def __init__(self, max_iterations: int = 10000):
        self.max_iterations = max_iterations
        self._geo_checker = PathConsistencyChecker(max_iterations)

    def check_consistency(self, network: FAQSRNetwork) -> ConsistencyResult:
        """
        Check consistency of an FA-QSR network.

        Args:
            network: The FA-QSR network to check

        Returns:
            ConsistencyResult with status and refined network
        """
        # Phase 1: Apply functional requirements to geometric constraints
        working = network.copy()

        for constraint in working.get_all_constraints():
            if constraint.functional:
                refined = constraint.apply_functional_requirements()
                if refined is None:
                    return ConsistencyResult(
                        status=ConsistencyStatus.INCONSISTENT,
                        conflict=f"Functional-geometric conflict for ({constraint.var1}, {constraint.var2})",
                        iterations=0,
                        variables_involved=(constraint.var1, constraint.var2)
                    )

                # Update geometric constraints with requirements
                working._geometric[(constraint.var1, constraint.var2)] = refined.geometric

        # Phase 2: Check geometric path consistency
        geo_network = working.to_rcc8_network()
        geo_result = self._geo_checker.check(geo_network)

        if not geo_result.is_consistent():
            return ConsistencyResult(
                status=ConsistencyStatus.INCONSISTENT,
                conflict=f"Geometric inconsistency: {geo_result.conflict}",
                iterations=geo_result.iterations,
                variables_involved=geo_result.variables_involved
            )

        # Phase 3: Check functional consistency (simplified)
        # Full functional consistency would require domain-specific reasoning
        func_result = self._check_functional_consistency(working)

        if not func_result.is_consistent():
            return func_result

        # Update working network with refined geometric constraints
        if geo_result.refined_network:
            for (v1, v2), rels in geo_result.refined_network._constraints.items():
                working._geometric[(v1, v2)] = rels

        return ConsistencyResult(
            status=ConsistencyStatus.CONSISTENT,
            refined_network=geo_result.refined_network,
            iterations=geo_result.iterations
        )

    def _check_functional_consistency(self, network: FAQSRNetwork) -> ConsistencyResult:
        """
        Check consistency of functional constraints.

        This is a simplified check that verifies:
        1. No contradictory functional relations (e.g., FSUPPORT and NO_FSUPPORT)
        2. Functional transitivity where applicable
        """
        # Check for contradictory functional relations
        for (v1, v2), func_rels in network._functional.items():
            # Check support contradictions
            has_support = any(
                r in FunctionalRelation.support_relations()
                for r in func_rels
            )
            has_no_support = FunctionalRelation.NO_FSUPPORT in func_rels

            if has_support and has_no_support:
                return ConsistencyResult(
                    status=ConsistencyStatus.INCONSISTENT,
                    conflict=f"Contradictory support relations for ({v1}, {v2})",
                    variables_involved=(v1, v2)
                )

            # Check containment contradictions
            has_contain = any(
                r in FunctionalRelation.containment_relations()
                for r in func_rels
            )
            has_no_contain = FunctionalRelation.NO_FCONTAIN in func_rels

            if has_contain and has_no_contain:
                return ConsistencyResult(
                    status=ConsistencyStatus.INCONSISTENT,
                    conflict=f"Contradictory containment relations for ({v1}, {v2})",
                    variables_involved=(v1, v2)
                )

        return ConsistencyResult(status=ConsistencyStatus.CONSISTENT)

    def infer_relations(self, network: FAQSRNetwork,
                       var1: str, var2: str) -> FAQSRConstraint:
        """
        Infer the possible relations between two variables.

        Uses constraint propagation to derive implied relations.

        Args:
            network: The FA-QSR network
            var1, var2: Variables to find relations between

        Returns:
            FAQSRConstraint with inferred relations
        """
        # Make path consistent first
        result = self.check_consistency(network)

        if not result.is_consistent():
            # Return empty constraint for inconsistent network
            return FAQSRConstraint(var1, var2, EMPTY, frozenset())

        if result.refined_network:
            geo_rels = result.refined_network.get_constraint(var1, var2)
        else:
            geo_rels = network.get_geometric_constraint(var1, var2)

        func_rels = network.get_functional_constraint(var1, var2)

        return FAQSRConstraint(var1, var2, geo_rels, func_rels)

    def minimal_network(self, network: FAQSRNetwork) -> FAQSRNetwork:
        """
        Compute the minimal network (tightest constraints).

        For each pair of variables, compute the tightest constraint
        that is entailed by the network.

        Warning: This can be expensive for large networks.
        """
        result = self.check_consistency(network)

        if not result.is_consistent():
            raise ValueError("Cannot compute minimal network for inconsistent network")

        working = network.copy()
        variables = list(working.variables)

        # For each pair, refine through all paths
        for i, vi in enumerate(variables):
            for j, vj in enumerate(variables):
                if i >= j:
                    continue

                current = working.get_geometric_constraint(vi, vj)

                # Refine through all intermediate variables
                for vk in variables:
                    if vk == vi or vk == vj:
                        continue

                    rik = working.get_geometric_constraint(vi, vk)
                    rkj = working.get_geometric_constraint(vk, vj)
                    composed = COMPOSITION_TABLE.compose_sets(rik, rkj)
                    current = current & composed

                working._geometric[(vi, vj)] = current

        return working


def demonstrate_reasoning():
    """Demonstrate the reasoning engine capabilities."""
    print("FA-QSR Reasoning Engine Demonstration")
    print("=" * 50)

    # Example 1: Simple RCC-8 network
    print("\n1. Pure RCC-8 Path Consistency")
    print("-" * 40)

    network1 = ConstraintNetwork()
    network1.add_variables("A", "B", "C")

    from qsr_base.rcc8 import relation_set
    # A is part of B
    network1.add_constraint("A", "B",
                           relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))
    # B is part of C
    network1.add_constraint("B", "C",
                           relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))

    checker = PathConsistencyChecker()
    result1 = checker.check(network1)

    print(f"Status: {result1.status.value}")
    print(f"Iterations: {result1.iterations}")

    if result1.refined_network:
        # Should infer A is part of C (transitivity)
        ac_rel = result1.refined_network.get_constraint("A", "C")
        print(f"Inferred A-C relation: {{{', '.join(str(r) for r in ac_rel)}}}")

    # Example 2: Inconsistent network
    print("\n2. Detecting Inconsistency")
    print("-" * 40)

    network2 = ConstraintNetwork()
    network2.add_variables("X", "Y", "Z")

    # X inside Y
    network2.add_constraint("X", "Y", relation_set(RCC8Relation.NTPP))
    # Y inside Z
    network2.add_constraint("Y", "Z", relation_set(RCC8Relation.NTPP))
    # X disconnected from Z (contradicts transitivity!)
    network2.add_constraint("X", "Z", relation_set(RCC8Relation.DC))

    result2 = checker.check(network2)

    print(f"Status: {result2.status.value}")
    if result2.conflict:
        print(f"Conflict: {result2.conflict}")

    # Example 3: FA-QSR hybrid reasoning
    print("\n3. FA-QSR Hybrid Reasoning")
    print("-" * 40)

    faqsr_network = FAQSRNetwork()
    faqsr_network.add_variable("mug", entity_type="container")
    faqsr_network.add_variable("hook", entity_type="support")
    faqsr_network.add_variable("wall", entity_type="surface")

    # Mug is hanging on hook (functional support)
    faqsr_network.add_functional_constraint(
        "mug", "hook",
        frozenset([FunctionalRelation.FSUPPORT_HANG])
    )

    # Hook is adhered to wall
    faqsr_network.add_functional_constraint(
        "hook", "wall",
        frozenset([FunctionalRelation.FSUPPORT_ADHERE])
    )

    reasoner = FAQSRReasoner()
    result3 = reasoner.check_consistency(faqsr_network)

    print(f"Status: {result3.status.value}")
    print(f"Network:\n{faqsr_network}")


if __name__ == "__main__":
    demonstrate_reasoning()
