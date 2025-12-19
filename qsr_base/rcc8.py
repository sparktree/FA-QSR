"""
RCC-8 (Region Connection Calculus) Implementation

This module implements the foundational RCC-8 calculus for qualitative spatial reasoning.
RCC-8 defines 8 jointly exhaustive and pairwise disjoint (JEPD) base relations between
spatial regions, along with composition tables for constraint propagation.

References:
- Randell, Cui & Cohn (1992) - A Spatial Logic based on Regions and Connection
- Renz & Nebel (1999) - On the Complexity of Qualitative Spatial Reasoning
- Cohn & Renz (2008) - Qualitative Spatial Representation and Reasoning

The 8 base relations are:
- DC: Disconnected
- EC: Externally Connected
- PO: Partial Overlap
- EQ: Equal
- TPP: Tangential Proper Part
- NTPP: Non-Tangential Proper Part
- TPPi: Tangential Proper Part inverse
- NTPPi: Non-Tangential Proper Part inverse
"""

from enum import Enum, auto
from typing import Set, Dict, Tuple, FrozenSet, Optional, List
from dataclasses import dataclass
from itertools import product


class RCC8Relation(Enum):
    """
    The 8 base relations of RCC-8.
    These are jointly exhaustive and pairwise disjoint (JEPD).
    """
    DC = auto()     # Disconnected
    EC = auto()     # Externally Connected
    PO = auto()     # Partial Overlap
    EQ = auto()     # Equal
    TPP = auto()    # Tangential Proper Part
    NTPP = auto()   # Non-Tangential Proper Part
    TPPi = auto()   # TPP inverse
    NTPPi = auto()  # NTPP inverse

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @classmethod
    def all_relations(cls) -> FrozenSet['RCC8Relation']:
        """Return all 8 base relations."""
        return frozenset(cls)

    def inverse(self) -> 'RCC8Relation':
        """Return the converse/inverse relation."""
        inverses = {
            RCC8Relation.DC: RCC8Relation.DC,
            RCC8Relation.EC: RCC8Relation.EC,
            RCC8Relation.PO: RCC8Relation.PO,
            RCC8Relation.EQ: RCC8Relation.EQ,
            RCC8Relation.TPP: RCC8Relation.TPPi,
            RCC8Relation.NTPP: RCC8Relation.NTPPi,
            RCC8Relation.TPPi: RCC8Relation.TPP,
            RCC8Relation.NTPPi: RCC8Relation.NTPP,
        }
        return inverses[self]


# Type alias for constraint sets (disjunctions of base relations)
RelationSet = FrozenSet[RCC8Relation]


def relation_set(*relations: RCC8Relation) -> RelationSet:
    """Create a relation set from given relations."""
    return frozenset(relations)


# Universal relation (any of the 8)
UNIVERSAL = frozenset(RCC8Relation)

# Empty relation (inconsistent)
EMPTY = frozenset()

# Common relation sets used in tractable fragments
BASIC = relation_set(RCC8Relation.DC, RCC8Relation.EC, RCC8Relation.PO,
                     RCC8Relation.EQ, RCC8Relation.TPP, RCC8Relation.NTPP,
                     RCC8Relation.TPPi, RCC8Relation.NTPPi)


class RCC8CompositionTable:
    """
    Composition table for RCC-8 relations.

    The composition r1 ; r2 gives the possible relations between X and Z
    when X r1 Y and Y r2 Z.
    """

    def __init__(self):
        self._table: Dict[Tuple[RCC8Relation, RCC8Relation], RelationSet] = {}
        self._build_table()

    def _build_table(self):
        """
        Build the complete RCC-8 composition table.
        Based on Randell et al. (1992) and standard RCC-8 references.
        """
        DC = RCC8Relation.DC
        EC = RCC8Relation.EC
        PO = RCC8Relation.PO
        EQ = RCC8Relation.EQ
        TPP = RCC8Relation.TPP
        NTPP = RCC8Relation.NTPP
        TPPi = RCC8Relation.TPPi
        NTPPi = RCC8Relation.NTPPi

        # Full RCC-8 composition table
        # Format: (r1, r2) -> possible relations when X r1 Y and Y r2 Z

        self._table = {
            # DC compositions
            (DC, DC): UNIVERSAL,
            (DC, EC): relation_set(DC, EC, PO, TPP, NTPP),
            (DC, PO): relation_set(DC, EC, PO, TPP, NTPP),
            (DC, EQ): relation_set(DC),
            (DC, TPP): relation_set(DC, EC, PO, TPP, NTPP),
            (DC, NTPP): relation_set(DC, EC, PO, TPP, NTPP),
            (DC, TPPi): relation_set(DC),
            (DC, NTPPi): relation_set(DC),

            # EC compositions
            (EC, DC): relation_set(DC, EC, PO, TPPi, NTPPi),
            (EC, EC): relation_set(DC, EC, PO, TPP, TPPi, EQ),
            (EC, PO): relation_set(DC, EC, PO, TPP, NTPP),
            (EC, EQ): relation_set(EC),
            (EC, TPP): relation_set(EC, PO, TPP, NTPP),
            (EC, NTPP): relation_set(PO, TPP, NTPP),
            (EC, TPPi): relation_set(DC, EC),
            (EC, NTPPi): relation_set(DC),

            # PO compositions
            (PO, DC): relation_set(DC, EC, PO, TPPi, NTPPi),
            (PO, EC): relation_set(DC, EC, PO, TPPi, NTPPi),
            (PO, PO): UNIVERSAL,
            (PO, EQ): relation_set(PO),
            (PO, TPP): relation_set(PO, TPP, NTPP),
            (PO, NTPP): relation_set(PO, TPP, NTPP),
            (PO, TPPi): relation_set(DC, EC, PO, TPPi, NTPPi),
            (PO, NTPPi): relation_set(DC, EC, PO, TPPi, NTPPi),

            # EQ compositions
            (EQ, DC): relation_set(DC),
            (EQ, EC): relation_set(EC),
            (EQ, PO): relation_set(PO),
            (EQ, EQ): relation_set(EQ),
            (EQ, TPP): relation_set(TPP),
            (EQ, NTPP): relation_set(NTPP),
            (EQ, TPPi): relation_set(TPPi),
            (EQ, NTPPi): relation_set(NTPPi),

            # TPP compositions
            (TPP, DC): relation_set(DC),
            (TPP, EC): relation_set(DC, EC),
            (TPP, PO): relation_set(DC, EC, PO, TPP, NTPP),
            (TPP, EQ): relation_set(TPP),
            (TPP, TPP): relation_set(TPP, NTPP),
            (TPP, NTPP): relation_set(NTPP),
            (TPP, TPPi): relation_set(DC, EC, PO, TPP, TPPi, EQ),
            (TPP, NTPPi): relation_set(DC, EC, PO, TPPi, NTPPi),

            # NTPP compositions
            (NTPP, DC): relation_set(DC),
            (NTPP, EC): relation_set(DC),
            (NTPP, PO): relation_set(DC, EC, PO, TPP, NTPP),
            (NTPP, EQ): relation_set(NTPP),
            (NTPP, TPP): relation_set(NTPP),
            (NTPP, NTPP): relation_set(NTPP),
            (NTPP, TPPi): relation_set(DC, EC, PO, TPP, NTPP),
            (NTPP, NTPPi): UNIVERSAL,

            # TPPi compositions
            (TPPi, DC): relation_set(DC, EC, PO, TPPi, NTPPi),
            (TPPi, EC): relation_set(EC, PO, TPPi, NTPPi),
            (TPPi, PO): relation_set(PO, TPPi, NTPPi),
            (TPPi, EQ): relation_set(TPPi),
            (TPPi, TPP): relation_set(PO, TPP, TPPi, EQ),
            (TPPi, NTPP): relation_set(PO, TPP, NTPP),
            (TPPi, TPPi): relation_set(TPPi, NTPPi),
            (TPPi, NTPPi): relation_set(NTPPi),

            # NTPPi compositions
            (NTPPi, DC): relation_set(DC, EC, PO, TPPi, NTPPi),
            (NTPPi, EC): relation_set(DC, EC, PO, TPPi, NTPPi),
            (NTPPi, PO): relation_set(DC, EC, PO, TPPi, NTPPi),
            (NTPPi, EQ): relation_set(NTPPi),
            (NTPPi, TPP): relation_set(DC, EC, PO, TPPi, NTPPi),
            (NTPPi, NTPP): UNIVERSAL,
            (NTPPi, TPPi): relation_set(NTPPi),
            (NTPPi, NTPPi): relation_set(NTPPi),
        }

    def compose(self, r1: RCC8Relation, r2: RCC8Relation) -> RelationSet:
        """
        Get the composition of two base relations.

        Args:
            r1: First relation (X r1 Y)
            r2: Second relation (Y r2 Z)

        Returns:
            Set of possible relations between X and Z
        """
        return self._table.get((r1, r2), EMPTY)

    def compose_sets(self, s1: RelationSet, s2: RelationSet) -> RelationSet:
        """
        Get the composition of two relation sets (weak composition).

        Args:
            s1: First relation set
            s2: Second relation set

        Returns:
            Set of possible relations (union of all compositions)
        """
        result: Set[RCC8Relation] = set()
        for r1, r2 in product(s1, s2):
            result.update(self.compose(r1, r2))
        return frozenset(result)


# Global composition table instance
COMPOSITION_TABLE = RCC8CompositionTable()


@dataclass(frozen=True)
class Constraint:
    """
    A constraint between two spatial variables.

    Attributes:
        var1: First variable name
        var2: Second variable name
        relations: Set of possible relations (disjunction)
    """
    var1: str
    var2: str
    relations: RelationSet

    def __post_init__(self):
        # Ensure var1 < var2 for canonical form
        if self.var1 > self.var2:
            object.__setattr__(self, 'var1', self.var2)
            object.__setattr__(self, 'var2', self.var1)
            # Also inverse the relations
            inversed = frozenset(r.inverse() for r in self.relations)
            object.__setattr__(self, 'relations', inversed)

    def intersect(self, other: 'Constraint') -> Optional['Constraint']:
        """Intersect this constraint with another (same variables)."""
        if self.var1 != other.var1 or self.var2 != other.var2:
            raise ValueError("Cannot intersect constraints on different variables")
        new_relations = self.relations & other.relations
        if not new_relations:
            return None  # Inconsistent
        return Constraint(self.var1, self.var2, new_relations)

    def is_universal(self) -> bool:
        """Check if this is the universal (uninformative) constraint."""
        return self.relations == UNIVERSAL

    def is_empty(self) -> bool:
        """Check if this constraint is empty (inconsistent)."""
        return len(self.relations) == 0

    def is_singleton(self) -> bool:
        """Check if this constraint specifies exactly one relation."""
        return len(self.relations) == 1

    def __repr__(self) -> str:
        rel_str = '{' + ', '.join(str(r) for r in sorted(self.relations, key=lambda x: x.name)) + '}'
        return f"{self.var1} {rel_str} {self.var2}"


class ConstraintNetwork:
    """
    A constraint network for RCC-8 qualitative spatial reasoning.

    Stores constraints between spatial variables and supports
    consistency checking and constraint propagation.
    """

    def __init__(self):
        self._variables: Set[str] = set()
        self._constraints: Dict[Tuple[str, str], RelationSet] = {}

    @property
    def variables(self) -> Set[str]:
        """Get all variables in the network."""
        return self._variables.copy()

    @property
    def num_variables(self) -> int:
        """Get the number of variables."""
        return len(self._variables)

    def add_variable(self, name: str) -> None:
        """Add a spatial variable to the network."""
        self._variables.add(name)

    def add_variables(self, *names: str) -> None:
        """Add multiple spatial variables."""
        self._variables.update(names)

    def add_constraint(self, var1: str, var2: str, relations: RelationSet) -> bool:
        """
        Add or refine a constraint between two variables.

        Args:
            var1: First variable
            var2: Second variable
            relations: Set of possible relations

        Returns:
            True if constraint was added/refined successfully,
            False if it results in inconsistency
        """
        # Ensure variables exist
        self._variables.add(var1)
        self._variables.add(var2)

        # Canonical ordering
        if var1 > var2:
            var1, var2 = var2, var1
            relations = frozenset(r.inverse() for r in relations)

        key = (var1, var2)

        # Intersect with existing constraint
        if key in self._constraints:
            relations = self._constraints[key] & relations
            if not relations:
                return False  # Inconsistent

        self._constraints[key] = relations
        return True

    def get_constraint(self, var1: str, var2: str) -> RelationSet:
        """
        Get the constraint between two variables.

        Returns UNIVERSAL if no specific constraint exists.
        """
        if var1 > var2:
            var1, var2 = var2, var1
            key = (var1, var2)
            if key in self._constraints:
                return frozenset(r.inverse() for r in self._constraints[key])
            return UNIVERSAL

        key = (var1, var2)
        return self._constraints.get(key, UNIVERSAL)

    def get_all_constraints(self) -> List[Constraint]:
        """Get all explicit constraints in the network."""
        return [
            Constraint(var1, var2, rels)
            for (var1, var2), rels in self._constraints.items()
        ]

    def copy(self) -> 'ConstraintNetwork':
        """Create a deep copy of this network."""
        new_network = ConstraintNetwork()
        new_network._variables = self._variables.copy()
        new_network._constraints = self._constraints.copy()
        return new_network

    def is_trivially_consistent(self) -> bool:
        """Quick check for obvious inconsistencies."""
        for rels in self._constraints.values():
            if not rels:
                return False
        return True

    def __repr__(self) -> str:
        lines = [f"ConstraintNetwork with {len(self._variables)} variables:"]
        for (v1, v2), rels in sorted(self._constraints.items()):
            rel_str = '{' + ', '.join(str(r) for r in sorted(rels, key=lambda x: x.name)) + '}'
            lines.append(f"  {v1} {rel_str} {v2}")
        return '\n'.join(lines)


# Tractable fragments of RCC-8
class TractableFragments:
    """
    Known tractable fragments of RCC-8.

    Based on Renz & Nebel (1999) analysis of maximal tractable subsets.
    """

    # The 3 maximal tractable subsets of RCC-8
    @staticmethod
    def H8() -> Set[RelationSet]:
        """
        The H8 (Horn) tractable fragment.
        Contains 148 relations, including all base relations.
        """
        DC = RCC8Relation.DC
        EC = RCC8Relation.EC
        PO = RCC8Relation.PO
        EQ = RCC8Relation.EQ
        TPP = RCC8Relation.TPP
        NTPP = RCC8Relation.NTPP
        TPPi = RCC8Relation.TPPi
        NTPPi = RCC8Relation.NTPPi

        # Core tractable relations (simplified set for demonstration)
        # Full H8 has 148 relations
        return {
            frozenset([DC]),
            frozenset([EC]),
            frozenset([PO]),
            frozenset([EQ]),
            frozenset([TPP]),
            frozenset([NTPP]),
            frozenset([TPPi]),
            frozenset([NTPPi]),
            frozenset([DC, EC]),
            frozenset([TPP, NTPP]),
            frozenset([TPPi, NTPPi]),
            frozenset([TPP, TPPi, EQ]),
            frozenset([DC, EC, PO, TPP, NTPP]),
            frozenset([DC, EC, PO, TPPi, NTPPi]),
            UNIVERSAL,
        }

    @staticmethod
    def is_in_tractable_fragment(relations: RelationSet) -> bool:
        """
        Check if a relation set is in a known tractable fragment.

        This is a simplified check - full analysis requires
        checking against all 148 H8 relations.
        """
        # Singleton base relations are always tractable
        if len(relations) == 1:
            return True

        # Universal is tractable
        if relations == UNIVERSAL:
            return True

        # Some known tractable combinations
        DC = RCC8Relation.DC
        EC = RCC8Relation.EC
        PO = RCC8Relation.PO
        EQ = RCC8Relation.EQ
        TPP = RCC8Relation.TPP
        NTPP = RCC8Relation.NTPP
        TPPi = RCC8Relation.TPPi
        NTPPi = RCC8Relation.NTPPi

        tractable_sets = {
            frozenset([DC, EC]),
            frozenset([TPP, NTPP]),
            frozenset([TPPi, NTPPi]),
            frozenset([TPP, EQ]),
            frozenset([TPPi, EQ]),
            frozenset([TPP, TPPi, EQ]),
            frozenset([DC, EC, PO]),
            frozenset([PO, TPP, NTPP]),
            frozenset([PO, TPPi, NTPPi]),
        }

        return relations in tractable_sets


def semantic_relations() -> Dict[str, RelationSet]:
    """
    Return semantically meaningful relation combinations
    commonly used in spatial language.
    """
    DC = RCC8Relation.DC
    EC = RCC8Relation.EC
    PO = RCC8Relation.PO
    EQ = RCC8Relation.EQ
    TPP = RCC8Relation.TPP
    NTPP = RCC8Relation.NTPP
    TPPi = RCC8Relation.TPPi
    NTPPi = RCC8Relation.NTPPi

    return {
        # Geometric 'in'/'inside' - proper part
        'inside': frozenset([TPP, NTPP]),

        # Geometric 'contains' - inverse of inside
        'contains': frozenset([TPPi, NTPPi]),

        # 'on' (touching exterior) - external connection or partial overlap
        'on_exterior': frozenset([EC]),

        # 'on' (on top of, partial) - can include overlap
        'on_surface': frozenset([EC, PO]),

        # 'outside' / 'separate from'
        'outside': frozenset([DC]),

        # 'near' / 'close to' - not overlapping but connected or close
        'near': frozenset([DC, EC]),

        # 'overlaps' / 'intersects'
        'overlaps': frozenset([PO]),

        # 'same as' / 'coincident'
        'coincident': frozenset([EQ]),

        # 'part of' - any parthood
        'part_of': frozenset([TPP, NTPP, EQ]),

        # 'disjoint from'
        'disjoint': frozenset([DC]),

        # 'connected to' - any connection
        'connected': frozenset([EC, PO, TPP, NTPP, TPPi, NTPPi, EQ]),

        # 'touching' - external connection only
        'touching': frozenset([EC]),
    }


if __name__ == "__main__":
    # Example usage
    print("RCC-8 Qualitative Spatial Reasoning Module")
    print("=" * 50)

    # Create a constraint network
    network = ConstraintNetwork()
    network.add_variables("room", "table", "cup", "coffee")

    # Add constraints
    network.add_constraint("table", "room", relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))
    network.add_constraint("cup", "table", relation_set(RCC8Relation.EC))
    network.add_constraint("coffee", "cup", relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))

    print("\nConstraint Network:")
    print(network)

    # Test composition
    print("\nComposition Examples:")
    comp = COMPOSITION_TABLE.compose(RCC8Relation.TPP, RCC8Relation.TPP)
    print(f"TPP ; TPP = {{{', '.join(str(r) for r in comp)}}}")

    comp = COMPOSITION_TABLE.compose(RCC8Relation.NTPP, RCC8Relation.NTPPi)
    print(f"NTPP ; NTPPi = {{{', '.join(str(r) for r in comp)}}}")

    # Semantic relations
    print("\nSemantic Relation Mappings:")
    for name, rels in semantic_relations().items():
        print(f"  {name}: {{{', '.join(str(r) for r in rels)}}}")
