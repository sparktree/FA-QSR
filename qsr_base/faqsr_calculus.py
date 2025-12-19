"""
FA-QSR Calculus - Functionally-Augmented Qualitative Spatial Reasoning

This module extends RCC-8 with functional spatial primitives (fsupport, fcontainment)
aligned with GUM's FunctionalSpatialModality categories. It implements weak-composition
tables for hybrid geometric-functional reasoning.

Key concepts:
- Functional Support (fsupport): Goal-conditioned sustainment
- Functional Containment (fcontainment): Purpose-driven enclosure
- Hybrid constraints combining geometric and functional requirements

References:
- Bateman et al. (2010) - GUM Spatial Extension
- Herskovits (1986) - Functional geometry in spatial language
- Coventry et al. (2001) - Functional determinants of spatial prepositions
"""

from enum import Enum, auto
from typing import Set, Dict, Tuple, FrozenSet, Optional, List, Union
from dataclasses import dataclass, field
from itertools import product

from .rcc8 import (
    RCC8Relation,
    RelationSet,
    relation_set,
    UNIVERSAL,
    EMPTY,
    COMPOSITION_TABLE,
    ConstraintNetwork as RCC8Network,
)


class FunctionalRelation(Enum):
    """
    Functional spatial relations that augment geometric RCC-8 relations.
    These encode affordances, force-dynamics, and pragmatic function.
    """
    # Functional Support variants
    FSUPPORT = auto()           # Generic functional support
    FSUPPORT_HANG = auto()      # Hanging support (hook, hanger)
    FSUPPORT_ADHERE = auto()    # Adhesive support (magnet, tape)
    FSUPPORT_ENGAGE = auto()    # Engagement support (peg, slot)
    FSUPPORT_BUOY = auto()      # Buoyancy support (floating)

    # Functional Containment variants
    FCONTAIN = auto()           # Generic functional containment
    FCONTAIN_PARTIAL = auto()   # Partial containment (vase, cup)
    FCONTAIN_PERMEABLE = auto() # Permeable containment (net, cage)
    FCONTAIN_BOUNDARY = auto()  # Boundary containment (fence, border)

    # Negated/Denial relations
    NO_FSUPPORT = auto()        # Denial of functional support
    NO_FCONTAIN = auto()        # Denial of functional containment

    # Access relations
    FACCESS = auto()            # Functional access
    FBLOCK = auto()             # Functional blocking

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @classmethod
    def support_relations(cls) -> FrozenSet['FunctionalRelation']:
        """Return all support-type functional relations."""
        return frozenset([
            cls.FSUPPORT, cls.FSUPPORT_HANG, cls.FSUPPORT_ADHERE,
            cls.FSUPPORT_ENGAGE, cls.FSUPPORT_BUOY
        ])

    @classmethod
    def containment_relations(cls) -> FrozenSet['FunctionalRelation']:
        """Return all containment-type functional relations."""
        return frozenset([
            cls.FCONTAIN, cls.FCONTAIN_PARTIAL,
            cls.FCONTAIN_PERMEABLE, cls.FCONTAIN_BOUNDARY
        ])


# Type aliases
FunctionalRelationSet = FrozenSet[FunctionalRelation]


@dataclass(frozen=True)
class FAQSRRelation:
    """
    A hybrid FA-QSR relation combining geometric and functional components.

    The geometric component specifies RCC-8 topological relations.
    The functional component specifies force-dynamic/affordance relations.
    Both must be satisfied for the relation to hold.
    """
    geometric: RelationSet = field(default_factory=lambda: UNIVERSAL)
    functional: FunctionalRelationSet = field(default_factory=frozenset)

    def is_purely_geometric(self) -> bool:
        """Check if this is a pure geometric constraint (no functional)."""
        return len(self.functional) == 0

    def is_purely_functional(self) -> bool:
        """Check if this is a pure functional constraint (universal geometric)."""
        return self.geometric == UNIVERSAL and len(self.functional) > 0

    def is_hybrid(self) -> bool:
        """Check if this combines geometric and functional constraints."""
        return self.geometric != UNIVERSAL and len(self.functional) > 0

    def __repr__(self) -> str:
        geo_str = '{' + ', '.join(str(r) for r in sorted(self.geometric, key=lambda x: x.name)) + '}'
        if self.functional:
            func_str = '{' + ', '.join(str(r) for r in sorted(self.functional, key=lambda x: x.name)) + '}'
            return f"FAQSR({geo_str}, {func_str})"
        return f"FAQSR({geo_str})"


class GeometricRequirements:
    """
    Defines the geometric prerequisites for functional relations.

    Functional relations have implicit geometric requirements - e.g.,
    fsupport requires some form of contact (EC, PO, TPP, etc.).
    """

    # Geometric requirements for functional relations
    REQUIREMENTS: Dict[FunctionalRelation, RelationSet] = {
        # Support requires contact or containment
        FunctionalRelation.FSUPPORT: relation_set(
            RCC8Relation.EC, RCC8Relation.PO,
            RCC8Relation.TPP, RCC8Relation.NTPP,
            RCC8Relation.TPPi, RCC8Relation.NTPPi
        ),
        FunctionalRelation.FSUPPORT_HANG: relation_set(
            RCC8Relation.EC, RCC8Relation.PO
        ),
        FunctionalRelation.FSUPPORT_ADHERE: relation_set(
            RCC8Relation.EC, RCC8Relation.PO
        ),
        FunctionalRelation.FSUPPORT_ENGAGE: relation_set(
            RCC8Relation.EC, RCC8Relation.PO,
            RCC8Relation.TPP, RCC8Relation.NTPP
        ),
        FunctionalRelation.FSUPPORT_BUOY: relation_set(
            RCC8Relation.EC, RCC8Relation.PO,
            RCC8Relation.TPPi, RCC8Relation.NTPPi
        ),

        # Containment requires overlap or parthood
        FunctionalRelation.FCONTAIN: relation_set(
            RCC8Relation.PO,
            RCC8Relation.TPPi, RCC8Relation.NTPPi
        ),
        FunctionalRelation.FCONTAIN_PARTIAL: relation_set(
            RCC8Relation.PO,
            RCC8Relation.TPPi, RCC8Relation.NTPPi
        ),
        FunctionalRelation.FCONTAIN_PERMEABLE: relation_set(
            RCC8Relation.PO,
            RCC8Relation.TPPi, RCC8Relation.NTPPi
        ),
        FunctionalRelation.FCONTAIN_BOUNDARY: relation_set(
            RCC8Relation.EC, RCC8Relation.PO,
            RCC8Relation.TPPi, RCC8Relation.NTPPi
        ),

        # Denial relations are compatible with any geometry
        FunctionalRelation.NO_FSUPPORT: UNIVERSAL,
        FunctionalRelation.NO_FCONTAIN: UNIVERSAL,

        # Access relations
        FunctionalRelation.FACCESS: relation_set(
            RCC8Relation.EC, RCC8Relation.PO,
            RCC8Relation.TPP, RCC8Relation.NTPP
        ),
        FunctionalRelation.FBLOCK: relation_set(
            RCC8Relation.EC, RCC8Relation.PO,
            RCC8Relation.TPP, RCC8Relation.NTPP,
            RCC8Relation.TPPi, RCC8Relation.NTPPi
        ),
    }

    @classmethod
    def get_requirements(cls, func_rel: FunctionalRelation) -> RelationSet:
        """Get the geometric requirements for a functional relation."""
        return cls.REQUIREMENTS.get(func_rel, UNIVERSAL)

    @classmethod
    def is_compatible(cls, func_rel: FunctionalRelation,
                     geo_rel: RCC8Relation) -> bool:
        """Check if a geometric relation is compatible with a functional one."""
        required = cls.get_requirements(func_rel)
        return geo_rel in required


class FunctionalCompositionTable:
    """
    Weak composition table for functional relations.

    Defines what functional relations can hold between X and Z
    given functional relations between X-Y and Y-Z.
    """

    def __init__(self):
        self._table: Dict[Tuple[FunctionalRelation, FunctionalRelation],
                         FunctionalRelationSet] = {}
        self._build_table()

    def _build_table(self):
        """Build the functional composition table."""
        FSUP = FunctionalRelation.FSUPPORT
        FCON = FunctionalRelation.FCONTAIN
        NO_SUP = FunctionalRelation.NO_FSUPPORT
        NO_CON = FunctionalRelation.NO_FCONTAIN

        # Functional support is not transitive in general
        # If A supports B and B supports C, A may or may not support C
        self._table = {
            # Support compositions
            (FSUP, FSUP): frozenset([FSUP, NO_SUP]),  # Not transitive
            (FSUP, FCON): frozenset([FCON, NO_CON]),  # If A supports B and B contains C
            (FSUP, NO_SUP): frozenset([NO_SUP]),
            (FSUP, NO_CON): frozenset([NO_CON]),

            # Containment compositions
            (FCON, FSUP): frozenset([FSUP, NO_SUP]),
            (FCON, FCON): frozenset([FCON]),  # Containment IS transitive
            (FCON, NO_SUP): frozenset([NO_SUP]),
            (FCON, NO_CON): frozenset([NO_CON]),

            # Denial compositions
            (NO_SUP, FSUP): frozenset([FSUP, NO_SUP]),
            (NO_SUP, FCON): frozenset([FCON, NO_CON]),
            (NO_SUP, NO_SUP): frozenset([NO_SUP]),
            (NO_SUP, NO_CON): frozenset([NO_CON]),

            (NO_CON, FSUP): frozenset([FSUP, NO_SUP]),
            (NO_CON, FCON): frozenset([FCON, NO_CON]),
            (NO_CON, NO_SUP): frozenset([NO_SUP]),
            (NO_CON, NO_CON): frozenset([NO_CON]),
        }

    def compose(self, f1: FunctionalRelation,
               f2: FunctionalRelation) -> FunctionalRelationSet:
        """Get composition of two functional relations."""
        return self._table.get((f1, f2), frozenset())


FUNCTIONAL_COMPOSITION = FunctionalCompositionTable()


@dataclass(frozen=True)
class FAQSRConstraint:
    """
    A constraint in the FA-QSR framework combining geometric and functional aspects.
    """
    var1: str
    var2: str
    geometric: RelationSet = field(default_factory=lambda: UNIVERSAL)
    functional: FunctionalRelationSet = field(default_factory=frozenset)

    def __post_init__(self):
        # Canonical ordering
        if self.var1 > self.var2:
            object.__setattr__(self, 'var1', self.var2)
            object.__setattr__(self, 'var2', self.var1)
            # Inverse geometric relations
            inversed_geo = frozenset(r.inverse() for r in self.geometric)
            object.__setattr__(self, 'geometric', inversed_geo)
            # Note: functional relations may need special inverse handling

    def intersect_geometric(self, other_geo: RelationSet) -> Optional['FAQSRConstraint']:
        """Intersect with additional geometric constraints."""
        new_geo = self.geometric & other_geo
        if not new_geo:
            return None
        return FAQSRConstraint(self.var1, self.var2, new_geo, self.functional)

    def apply_functional_requirements(self) -> Optional['FAQSRConstraint']:
        """
        Apply geometric requirements implied by functional relations.
        Returns None if inconsistent.
        """
        if not self.functional:
            return self

        # Collect all geometric requirements from functional relations
        required_geo: Set[RCC8Relation] = set()
        first = True
        for func_rel in self.functional:
            func_requirements = GeometricRequirements.get_requirements(func_rel)
            if first:
                required_geo = set(func_requirements)
                first = False
            else:
                required_geo &= func_requirements

        if not required_geo:
            return None  # Inconsistent functional requirements

        # Intersect with existing geometric constraints
        new_geo = self.geometric & frozenset(required_geo)
        if not new_geo:
            return None  # Geometric-functional inconsistency

        return FAQSRConstraint(self.var1, self.var2, new_geo, self.functional)

    def is_consistent(self) -> bool:
        """Check if this constraint is potentially satisfiable."""
        if not self.geometric:
            return False
        refined = self.apply_functional_requirements()
        return refined is not None

    def complexity_class(self) -> str:
        """
        Estimate the computational complexity class of this constraint.

        Returns:
            'tractable': Pure geometric in known tractable fragment
            'np-hard': Hybrid or complex functional
            'unknown': Needs further analysis
        """
        if not self.functional:
            # Pure geometric - check tractable fragments
            from .rcc8 import TractableFragments
            if TractableFragments.is_in_tractable_fragment(self.geometric):
                return 'tractable'
            return 'unknown'

        # Hybrid constraints are generally NP-hard
        return 'np-hard'

    def __repr__(self) -> str:
        geo_str = '{' + ', '.join(str(r) for r in sorted(self.geometric, key=lambda x: x.name)) + '}'
        if self.functional:
            func_str = '{' + ', '.join(str(r) for r in sorted(self.functional, key=lambda x: x.name)) + '}'
            return f"{self.var1} [{geo_str}; {func_str}] {self.var2}"
        return f"{self.var1} [{geo_str}] {self.var2}"


class FAQSRNetwork:
    """
    A constraint network for Functionally-Augmented QSR.

    Extends RCC-8 networks with functional spatial constraints,
    supporting hybrid geometric-functional reasoning.
    """

    def __init__(self):
        self._variables: Set[str] = set()
        self._geometric: Dict[Tuple[str, str], RelationSet] = {}
        self._functional: Dict[Tuple[str, str], FunctionalRelationSet] = {}
        self._entity_types: Dict[str, str] = {}  # Variable -> type mapping
        self._entity_affordances: Dict[str, Set[str]] = {}  # Variable -> affordances

    @property
    def variables(self) -> Set[str]:
        """Get all variables in the network."""
        return self._variables.copy()

    @property
    def num_variables(self) -> int:
        return len(self._variables)

    def add_variable(self, name: str, entity_type: Optional[str] = None,
                    affordances: Optional[Set[str]] = None) -> None:
        """
        Add a spatial variable with optional type and affordance information.

        Args:
            name: Variable name
            entity_type: Semantic type (e.g., 'container', 'surface', 'hook')
            affordances: Set of affordance labels
        """
        self._variables.add(name)
        if entity_type:
            self._entity_types[name] = entity_type
        if affordances:
            self._entity_affordances[name] = affordances

    def _normalize_key(self, var1: str, var2: str) -> Tuple[str, str]:
        """Get canonical key ordering."""
        if var1 > var2:
            return (var2, var1)
        return (var1, var2)

    def add_geometric_constraint(self, var1: str, var2: str,
                                relations: RelationSet) -> bool:
        """
        Add or refine a geometric constraint.

        Returns False if this creates an inconsistency.
        """
        self._variables.add(var1)
        self._variables.add(var2)

        key = self._normalize_key(var1, var2)
        if var1 > var2:
            relations = frozenset(r.inverse() for r in relations)

        if key in self._geometric:
            relations = self._geometric[key] & relations
            if not relations:
                return False

        self._geometric[key] = relations
        return True

    def add_functional_constraint(self, var1: str, var2: str,
                                  relations: FunctionalRelationSet) -> bool:
        """
        Add functional constraint(s) between variables.

        Also enforces geometric requirements of the functional relations.
        Returns False if inconsistent.
        """
        self._variables.add(var1)
        self._variables.add(var2)

        key = self._normalize_key(var1, var2)

        # Add to functional constraints
        if key in self._functional:
            self._functional[key] = self._functional[key] | relations
        else:
            self._functional[key] = relations

        # Enforce geometric requirements
        for func_rel in relations:
            geo_req = GeometricRequirements.get_requirements(func_rel)
            if not self.add_geometric_constraint(var1, var2, geo_req):
                return False

        return True

    def add_constraint(self, constraint: FAQSRConstraint) -> bool:
        """Add a full FA-QSR constraint."""
        if constraint.geometric:
            if not self.add_geometric_constraint(
                constraint.var1, constraint.var2, constraint.geometric
            ):
                return False

        if constraint.functional:
            if not self.add_functional_constraint(
                constraint.var1, constraint.var2, constraint.functional
            ):
                return False

        return True

    def get_geometric_constraint(self, var1: str, var2: str) -> RelationSet:
        """Get geometric constraint between two variables."""
        key = self._normalize_key(var1, var2)
        rels = self._geometric.get(key, UNIVERSAL)
        if var1 > var2:
            return frozenset(r.inverse() for r in rels)
        return rels

    def get_functional_constraint(self, var1: str, var2: str) -> FunctionalRelationSet:
        """Get functional constraints between two variables."""
        key = self._normalize_key(var1, var2)
        return self._functional.get(key, frozenset())

    def get_constraint(self, var1: str, var2: str) -> FAQSRConstraint:
        """Get full FA-QSR constraint between two variables."""
        return FAQSRConstraint(
            var1, var2,
            self.get_geometric_constraint(var1, var2),
            self.get_functional_constraint(var1, var2)
        )

    def get_all_constraints(self) -> List[FAQSRConstraint]:
        """Get all explicit constraints in the network."""
        constraints = []
        all_keys = set(self._geometric.keys()) | set(self._functional.keys())
        for var1, var2 in all_keys:
            constraints.append(self.get_constraint(var1, var2))
        return constraints

    def to_rcc8_network(self) -> RCC8Network:
        """
        Extract the geometric component as an RCC-8 network.

        Useful for applying standard RCC-8 reasoning algorithms.
        """
        network = RCC8Network()
        for var in self._variables:
            network.add_variable(var)
        for (v1, v2), rels in self._geometric.items():
            network.add_constraint(v1, v2, rels)
        return network

    def estimate_complexity(self) -> Dict[str, any]:
        """
        Analyze the computational complexity profile of this network.

        Returns dictionary with:
        - has_functional: Whether network has functional constraints
        - pure_geometric_fraction: Fraction of constraints that are pure geometric
        - estimated_class: Overall complexity estimate
        """
        total_constraints = len(self._geometric)
        functional_constraints = len(self._functional)

        has_functional = functional_constraints > 0
        pure_geo = total_constraints - functional_constraints

        if total_constraints == 0:
            pure_fraction = 1.0
        else:
            pure_fraction = pure_geo / total_constraints

        # Estimate complexity class
        if not has_functional:
            estimated = 'polynomial'
        elif pure_fraction > 0.8:
            estimated = 'likely_tractable'
        else:
            estimated = 'np-hard'

        return {
            'has_functional': has_functional,
            'total_constraints': total_constraints,
            'functional_constraints': functional_constraints,
            'pure_geometric_fraction': pure_fraction,
            'estimated_class': estimated,
        }

    def copy(self) -> 'FAQSRNetwork':
        """Create a deep copy of this network."""
        new_network = FAQSRNetwork()
        new_network._variables = self._variables.copy()
        new_network._geometric = self._geometric.copy()
        new_network._functional = self._functional.copy()
        new_network._entity_types = self._entity_types.copy()
        new_network._entity_affordances = {
            k: v.copy() for k, v in self._entity_affordances.items()
        }
        return new_network

    def __repr__(self) -> str:
        lines = [f"FAQSRNetwork with {len(self._variables)} variables:"]
        lines.append(f"  Complexity profile: {self.estimate_complexity()['estimated_class']}")

        for constraint in self.get_all_constraints():
            lines.append(f"  {constraint}")

        return '\n'.join(lines)


# Predefined FA-QSR relations for common spatial expressions
class CommonFAQSRPatterns:
    """
    Common FA-QSR constraint patterns for spatial language expressions.

    Maps natural language spatial descriptions to appropriate constraint sets.
    """

    @staticmethod
    def on_surface() -> FAQSRConstraint:
        """'X on Y' - horizontal support (table, shelf)"""
        return FAQSRConstraint(
            "trajector", "landmark",
            geometric=relation_set(RCC8Relation.EC, RCC8Relation.PO),
            functional=frozenset([FunctionalRelation.FSUPPORT])
        )

    @staticmethod
    def on_vertical() -> FAQSRConstraint:
        """'X on Y' - vertical support (wall, hook)"""
        return FAQSRConstraint(
            "trajector", "landmark",
            geometric=relation_set(RCC8Relation.EC, RCC8Relation.PO),
            functional=frozenset([FunctionalRelation.FSUPPORT_ADHERE])
        )

    @staticmethod
    def on_hanging() -> FAQSRConstraint:
        """'X on Y' - hanging support (hook, peg)"""
        return FAQSRConstraint(
            "trajector", "landmark",
            geometric=relation_set(RCC8Relation.EC, RCC8Relation.PO),
            functional=frozenset([FunctionalRelation.FSUPPORT_HANG])
        )

    @staticmethod
    def in_container() -> FAQSRConstraint:
        """'X in Y' - containment (vase, cup, box)"""
        return FAQSRConstraint(
            "trajector", "landmark",
            geometric=relation_set(RCC8Relation.TPP, RCC8Relation.NTPP, RCC8Relation.PO),
            functional=frozenset([FunctionalRelation.FCONTAIN])
        )

    @staticmethod
    def in_region() -> FAQSRConstraint:
        """'X in Y' - regional containment (room, city)"""
        return FAQSRConstraint(
            "trajector", "landmark",
            geometric=relation_set(RCC8Relation.TPP, RCC8Relation.NTPP),
            functional=frozenset([FunctionalRelation.FCONTAIN_BOUNDARY])
        )

    @staticmethod
    def in_net() -> FAQSRConstraint:
        """'X in Y' - permeable containment (net, cage)"""
        return FAQSRConstraint(
            "trajector", "landmark",
            geometric=relation_set(RCC8Relation.TPP, RCC8Relation.NTPP, RCC8Relation.PO),
            functional=frozenset([FunctionalRelation.FCONTAIN_PERMEABLE])
        )


if __name__ == "__main__":
    print("FA-QSR Calculus Module")
    print("=" * 50)

    # Create a network with functional constraints
    network = FAQSRNetwork()

    # Add variables with affordance information
    network.add_variable("vase", entity_type="container",
                        affordances={"containment", "support"})
    network.add_variable("flowers", entity_type="object")
    network.add_variable("table", entity_type="surface",
                        affordances={"support"})
    network.add_variable("room", entity_type="region")

    # Add constraints
    # "The flowers are in the vase" - functional containment
    network.add_functional_constraint(
        "flowers", "vase",
        frozenset([FunctionalRelation.FCONTAIN_PARTIAL])
    )

    # "The vase is on the table" - functional support
    network.add_functional_constraint(
        "vase", "table",
        frozenset([FunctionalRelation.FSUPPORT])
    )

    # "The table is in the room" - geometric containment
    network.add_geometric_constraint(
        "table", "room",
        relation_set(RCC8Relation.TPP, RCC8Relation.NTPP)
    )

    print("\nFA-QSR Network:")
    print(network)

    print("\nComplexity Analysis:")
    complexity = network.estimate_complexity()
    for key, value in complexity.items():
        print(f"  {key}: {value}")

    print("\nCommon FA-QSR Patterns:")
    print(f"  on_surface: {CommonFAQSRPatterns.on_surface()}")
    print(f"  in_container: {CommonFAQSRPatterns.in_container()}")
