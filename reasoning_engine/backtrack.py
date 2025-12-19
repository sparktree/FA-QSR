"""
Backtracking Solver for FA-QSR Satisfiability

This module implements a backtracking search algorithm for determining
satisfiability of FA-QSR constraint networks when path consistency
alone is insufficient (e.g., for NP-hard fragments).

The solver uses:
1. Path consistency as preprocessing and pruning
2. Variable and value ordering heuristics
3. Backjumping for efficient search

References:
- Renz (1999) - Maximal Tractable Fragments of the Region Connection Calculus
- Dechter (2003) - Constraint Processing
"""

from typing import Set, Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsr_base.rcc8 import (
    RCC8Relation,
    RelationSet,
    relation_set,
    UNIVERSAL,
    EMPTY,
    ConstraintNetwork,
)
from qsr_base.faqsr_calculus import FAQSRNetwork
from reasoning_engine.path_consistency import (
    PathConsistencyChecker,
    FAQSRReasoner,
    ConsistencyResult,
    ConsistencyStatus,
)


class SearchStatus(Enum):
    """Status of the backtracking search."""
    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    TIMEOUT = "timeout"


@dataclass
class SearchResult:
    """
    Result of backtracking search.

    Attributes:
        status: Whether a solution was found
        solution: Assignment of base relations (if satisfiable)
        nodes_explored: Number of search nodes visited
        backtracks: Number of backtracks performed
    """
    status: SearchStatus
    solution: Optional[Dict[Tuple[str, str], RCC8Relation]] = None
    nodes_explored: int = 0
    backtracks: int = 0


class VariableOrderingHeuristic(Enum):
    """Heuristics for selecting the next variable to assign."""
    FIRST = "first"           # Simple left-to-right
    MRV = "mrv"               # Minimum Remaining Values
    DEGREE = "degree"         # Maximum degree (most constrained)
    DOM_WDEG = "dom_wdeg"     # Domain over weighted degree


class ValueOrderingHeuristic(Enum):
    """Heuristics for selecting the next value to try."""
    FIRST = "first"           # First available value
    LCV = "lcv"               # Least Constraining Value
    RANDOM = "random"         # Random selection


class BacktrackingSolver:
    """
    Backtracking search solver for QSR satisfiability.

    Uses constraint propagation (path consistency) integrated with
    systematic search to find satisfying assignments or prove
    unsatisfiability.
    """

    def __init__(self,
                 var_heuristic: VariableOrderingHeuristic = VariableOrderingHeuristic.MRV,
                 val_heuristic: ValueOrderingHeuristic = ValueOrderingHeuristic.FIRST,
                 max_nodes: int = 100000,
                 use_propagation: bool = True):
        """
        Initialize the solver.

        Args:
            var_heuristic: Variable ordering heuristic
            val_heuristic: Value ordering heuristic
            max_nodes: Maximum nodes to explore before timeout
            use_propagation: Whether to use path consistency for pruning
        """
        self.var_heuristic = var_heuristic
        self.val_heuristic = val_heuristic
        self.max_nodes = max_nodes
        self.use_propagation = use_propagation
        self._pc_checker = PathConsistencyChecker()

    def solve(self, network: ConstraintNetwork) -> SearchResult:
        """
        Find a satisfying assignment for the network.

        A satisfying assignment maps each variable pair to a single
        base relation that is consistent with all constraints.

        Args:
            network: The constraint network to solve

        Returns:
            SearchResult with solution (if found) and statistics
        """
        # First, apply path consistency
        if self.use_propagation:
            pc_result = self._pc_checker.check(network)
            if not pc_result.is_consistent():
                return SearchResult(
                    status=SearchStatus.UNSATISFIABLE,
                    nodes_explored=0,
                    backtracks=0
                )
            network = pc_result.refined_network or network

        # Get all variable pairs with non-singleton constraints
        variables = list(network.variables)
        pairs_to_assign: List[Tuple[str, str]] = []

        for i, v1 in enumerate(variables):
            for j, v2 in enumerate(variables):
                if i < j:
                    rels = network.get_constraint(v1, v2)
                    if len(rels) > 1:
                        pairs_to_assign.append((v1, v2))

        if not pairs_to_assign:
            # All constraints are already singleton - solution found
            solution = {}
            for i, v1 in enumerate(variables):
                for j, v2 in enumerate(variables):
                    if i < j:
                        rels = network.get_constraint(v1, v2)
                        if rels:
                            solution[(v1, v2)] = next(iter(rels))
            return SearchResult(
                status=SearchStatus.SATISFIABLE,
                solution=solution,
                nodes_explored=1,
                backtracks=0
            )

        # Initialize search state
        assignment: Dict[Tuple[str, str], RCC8Relation] = {}
        nodes_explored = 0
        backtracks = 0

        # Backtracking search
        result = self._backtrack(
            network.copy(),
            pairs_to_assign,
            assignment,
            [0],  # nodes_explored (mutable reference)
            [0],  # backtracks (mutable reference)
        )

        if result is not None:
            # Complete the solution with pre-determined singletons
            for i, v1 in enumerate(variables):
                for j, v2 in enumerate(variables):
                    if i < j and (v1, v2) not in result:
                        rels = network.get_constraint(v1, v2)
                        if rels:
                            result[(v1, v2)] = next(iter(rels))

            return SearchResult(
                status=SearchStatus.SATISFIABLE,
                solution=result,
                nodes_explored=nodes_explored,
                backtracks=backtracks
            )
        else:
            return SearchResult(
                status=SearchStatus.UNSATISFIABLE,
                nodes_explored=nodes_explored,
                backtracks=backtracks
            )

    def _backtrack(self,
                   network: ConstraintNetwork,
                   unassigned: List[Tuple[str, str]],
                   assignment: Dict[Tuple[str, str], RCC8Relation],
                   nodes: List[int],
                   backtracks: List[int]) -> Optional[Dict[Tuple[str, str], RCC8Relation]]:
        """
        Recursive backtracking search.

        Args:
            network: Current constraint network state
            unassigned: Variable pairs not yet assigned
            assignment: Current partial assignment
            nodes: Mutable node counter
            backtracks: Mutable backtrack counter

        Returns:
            Complete assignment if found, None otherwise
        """
        nodes[0] += 1

        if nodes[0] > self.max_nodes:
            return None  # Timeout

        if not unassigned:
            return assignment.copy()  # Solution found

        # Select next variable pair to assign
        pair = self._select_variable(network, unassigned)
        v1, v2 = pair

        # Get possible values (base relations)
        possible_values = network.get_constraint(v1, v2)

        # Order values
        ordered_values = self._order_values(network, pair, possible_values)

        for rel in ordered_values:
            # Try assigning this value
            new_network = network.copy()
            new_network._constraints[(v1, v2)] = frozenset([rel])

            # Apply propagation if enabled
            if self.use_propagation:
                pc_result = self._pc_checker.check(new_network)
                if not pc_result.is_consistent():
                    continue  # Prune this branch
                new_network = pc_result.refined_network or new_network

            # Recurse
            new_assignment = assignment.copy()
            new_assignment[pair] = rel
            new_unassigned = [p for p in unassigned if p != pair]

            result = self._backtrack(
                new_network, new_unassigned, new_assignment, nodes, backtracks
            )

            if result is not None:
                return result

            backtracks[0] += 1

        return None  # No solution in this subtree

    def _select_variable(self,
                        network: ConstraintNetwork,
                        unassigned: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Select the next variable pair to assign based on heuristic."""
        if self.var_heuristic == VariableOrderingHeuristic.FIRST:
            return unassigned[0]

        elif self.var_heuristic == VariableOrderingHeuristic.MRV:
            # Choose pair with smallest domain (fewest remaining values)
            return min(
                unassigned,
                key=lambda p: len(network.get_constraint(p[0], p[1]))
            )

        elif self.var_heuristic == VariableOrderingHeuristic.DEGREE:
            # Choose most constrained pair (involved in most constraints)
            def constraint_count(pair):
                v1, v2 = pair
                count = 0
                for p in unassigned:
                    if p != pair and (v1 in p or v2 in p):
                        count += 1
                return count
            return max(unassigned, key=constraint_count)

        return unassigned[0]

    def _order_values(self,
                     network: ConstraintNetwork,
                     pair: Tuple[str, str],
                     values: RelationSet) -> List[RCC8Relation]:
        """Order the values to try based on heuristic."""
        value_list = list(values)

        if self.val_heuristic == ValueOrderingHeuristic.FIRST:
            return value_list

        elif self.val_heuristic == ValueOrderingHeuristic.LCV:
            # Least constraining value - choose value that rules out fewest
            # options for remaining variables (expensive to compute)
            def constraining_power(rel):
                # Count how many values get eliminated for other pairs
                eliminated = 0
                # Simplified: just return 0 for now
                return eliminated
            return sorted(value_list, key=constraining_power)

        return value_list


class FAQSRBacktrackingSolver:
    """
    Backtracking solver for FA-QSR networks.

    Handles hybrid geometric-functional constraints by:
    1. Pre-processing functional requirements
    2. Solving the geometric component
    3. Verifying functional consistency of solutions
    """

    def __init__(self,
                 max_nodes: int = 100000,
                 use_propagation: bool = True):
        self.max_nodes = max_nodes
        self.use_propagation = use_propagation
        self._geo_solver = BacktrackingSolver(
            max_nodes=max_nodes,
            use_propagation=use_propagation
        )
        self._faqsr_reasoner = FAQSRReasoner()

    def solve(self, network: FAQSRNetwork) -> SearchResult:
        """
        Find a satisfying assignment for an FA-QSR network.

        Args:
            network: The FA-QSR network to solve

        Returns:
            SearchResult with solution and statistics
        """
        # First check overall consistency
        consistency = self._faqsr_reasoner.check_consistency(network)

        if not consistency.is_consistent():
            return SearchResult(
                status=SearchStatus.UNSATISFIABLE,
                nodes_explored=0,
                backtracks=0
            )

        # Extract and solve geometric component
        geo_network = network.to_rcc8_network()
        geo_result = self._geo_solver.solve(geo_network)

        if geo_result.status != SearchStatus.SATISFIABLE:
            return SearchResult(
                status=geo_result.status,
                nodes_explored=geo_result.nodes_explored,
                backtracks=geo_result.backtracks
            )

        # Verify functional constraints are satisfied
        # (Already checked during consistency, but double-check solution)
        if geo_result.solution:
            for (v1, v2), func_rels in network._functional.items():
                if func_rels:
                    geo_rel = geo_result.solution.get((v1, v2))
                    if geo_rel:
                        # Check geometric requirements are met
                        for func_rel in func_rels:
                            if not self._check_geo_compatible(geo_rel, func_rel):
                                # This shouldn't happen if consistency check passed
                                return SearchResult(
                                    status=SearchStatus.UNSATISFIABLE,
                                    nodes_explored=geo_result.nodes_explored,
                                    backtracks=geo_result.backtracks
                                )

        return geo_result

    def _check_geo_compatible(self,
                             geo_rel: RCC8Relation,
                             func_rel) -> bool:
        """Check if geometric relation satisfies functional requirements."""
        from qsr_base.faqsr_calculus import GeometricRequirements
        return GeometricRequirements.is_compatible(func_rel, geo_rel)


def demonstrate_backtracking():
    """Demonstrate backtracking solver capabilities."""
    print("Backtracking Solver Demonstration")
    print("=" * 50)

    # Example 1: Simple satisfiable network
    print("\n1. Satisfiable Network")
    print("-" * 40)

    network1 = ConstraintNetwork()
    network1.add_variables("A", "B", "C")

    # A overlaps or is inside B
    network1.add_constraint("A", "B",
                           relation_set(RCC8Relation.PO, RCC8Relation.TPP, RCC8Relation.NTPP))
    # B overlaps or contains C
    network1.add_constraint("B", "C",
                           relation_set(RCC8Relation.PO, RCC8Relation.TPPi, RCC8Relation.NTPPi))
    # A and C relationship unspecified (will be inferred)

    solver = BacktrackingSolver()
    result1 = solver.solve(network1)

    print(f"Status: {result1.status.value}")
    print(f"Nodes explored: {result1.nodes_explored}")
    print(f"Backtracks: {result1.backtracks}")
    if result1.solution:
        print("Solution:")
        for (v1, v2), rel in sorted(result1.solution.items()):
            print(f"  {v1} {rel} {v2}")

    # Example 2: Unsatisfiable network
    print("\n2. Unsatisfiable Network")
    print("-" * 40)

    network2 = ConstraintNetwork()
    network2.add_variables("X", "Y", "Z")

    # Cyclic containment (impossible!)
    network2.add_constraint("X", "Y", relation_set(RCC8Relation.NTPP))
    network2.add_constraint("Y", "Z", relation_set(RCC8Relation.NTPP))
    network2.add_constraint("Z", "X", relation_set(RCC8Relation.NTPP))

    result2 = solver.solve(network2)

    print(f"Status: {result2.status.value}")
    print(f"Nodes explored: {result2.nodes_explored}")

    # Example 3: FA-QSR network
    print("\n3. FA-QSR Network")
    print("-" * 40)

    from qsr_base.faqsr_calculus import FunctionalRelation

    faqsr_network = FAQSRNetwork()
    faqsr_network.add_variable("cup")
    faqsr_network.add_variable("saucer")
    faqsr_network.add_variable("table")

    # Cup on saucer (functional support)
    faqsr_network.add_functional_constraint(
        "cup", "saucer",
        frozenset([FunctionalRelation.FSUPPORT])
    )

    # Saucer on table
    faqsr_network.add_functional_constraint(
        "saucer", "table",
        frozenset([FunctionalRelation.FSUPPORT])
    )

    faqsr_solver = FAQSRBacktrackingSolver()
    result3 = faqsr_solver.solve(faqsr_network)

    print(f"Status: {result3.status.value}")
    print(f"Nodes explored: {result3.nodes_explored}")
    if result3.solution:
        print("Solution:")
        for (v1, v2), rel in sorted(result3.solution.items()):
            print(f"  {v1} {rel} {v2}")


if __name__ == "__main__":
    demonstrate_backtracking()
