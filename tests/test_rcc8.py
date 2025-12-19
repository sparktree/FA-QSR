"""
Unit tests for RCC-8 module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from qsr_base.rcc8 import (
    RCC8Relation,
    RelationSet,
    relation_set,
    UNIVERSAL,
    EMPTY,
    COMPOSITION_TABLE,
    Constraint,
    ConstraintNetwork,
    TractableFragments,
)


class TestRCC8Relations(unittest.TestCase):
    """Test RCC-8 base relations."""

    def test_all_relations_count(self):
        """Test that we have exactly 8 base relations."""
        self.assertEqual(len(RCC8Relation), 8)

    def test_relation_inverses(self):
        """Test that inverses are correct."""
        # Symmetric relations
        self.assertEqual(RCC8Relation.DC.inverse(), RCC8Relation.DC)
        self.assertEqual(RCC8Relation.EC.inverse(), RCC8Relation.EC)
        self.assertEqual(RCC8Relation.PO.inverse(), RCC8Relation.PO)
        self.assertEqual(RCC8Relation.EQ.inverse(), RCC8Relation.EQ)

        # Asymmetric relations
        self.assertEqual(RCC8Relation.TPP.inverse(), RCC8Relation.TPPi)
        self.assertEqual(RCC8Relation.NTPP.inverse(), RCC8Relation.NTPPi)
        self.assertEqual(RCC8Relation.TPPi.inverse(), RCC8Relation.TPP)
        self.assertEqual(RCC8Relation.NTPPi.inverse(), RCC8Relation.NTPP)

    def test_double_inverse(self):
        """Test that inverse of inverse is identity."""
        for rel in RCC8Relation:
            self.assertEqual(rel.inverse().inverse(), rel)


class TestCompositionTable(unittest.TestCase):
    """Test RCC-8 composition table."""

    def test_identity_composition(self):
        """Test composition with EQ (identity)."""
        for rel in RCC8Relation:
            # r ; EQ = r
            result = COMPOSITION_TABLE.compose(rel, RCC8Relation.EQ)
            self.assertEqual(result, frozenset([rel]))

            # EQ ; r = r
            result = COMPOSITION_TABLE.compose(RCC8Relation.EQ, rel)
            self.assertEqual(result, frozenset([rel]))

    def test_tpp_transitivity(self):
        """Test TPP ; TPP includes NTPP (transitivity of proper part)."""
        result = COMPOSITION_TABLE.compose(RCC8Relation.TPP, RCC8Relation.TPP)
        self.assertIn(RCC8Relation.TPP, result)
        self.assertIn(RCC8Relation.NTPP, result)

    def test_ntpp_transitivity(self):
        """Test NTPP ; NTPP = NTPP (strict transitivity)."""
        result = COMPOSITION_TABLE.compose(RCC8Relation.NTPP, RCC8Relation.NTPP)
        self.assertEqual(result, frozenset([RCC8Relation.NTPP]))

    def test_set_composition(self):
        """Test composition of relation sets."""
        s1 = relation_set(RCC8Relation.TPP, RCC8Relation.NTPP)
        s2 = relation_set(RCC8Relation.EQ)
        result = COMPOSITION_TABLE.compose_sets(s1, s2)
        self.assertEqual(result, s1)


class TestConstraintNetwork(unittest.TestCase):
    """Test constraint network operations."""

    def test_empty_network(self):
        """Test empty network creation."""
        network = ConstraintNetwork()
        self.assertEqual(network.num_variables, 0)

    def test_add_variables(self):
        """Test adding variables."""
        network = ConstraintNetwork()
        network.add_variables("A", "B", "C")
        self.assertEqual(network.num_variables, 3)
        self.assertIn("A", network.variables)
        self.assertIn("B", network.variables)
        self.assertIn("C", network.variables)

    def test_add_constraint(self):
        """Test adding constraints."""
        network = ConstraintNetwork()
        network.add_variables("A", "B")

        success = network.add_constraint("A", "B",
                                        relation_set(RCC8Relation.TPP))
        self.assertTrue(success)

        result = network.get_constraint("A", "B")
        self.assertEqual(result, frozenset([RCC8Relation.TPP]))

    def test_constraint_refinement(self):
        """Test that adding constraints refines (intersects)."""
        network = ConstraintNetwork()
        network.add_variables("A", "B")

        # Add broad constraint
        network.add_constraint("A", "B",
                              relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))

        # Refine to single relation
        success = network.add_constraint("A", "B",
                                        relation_set(RCC8Relation.TPP))
        self.assertTrue(success)

        result = network.get_constraint("A", "B")
        self.assertEqual(result, frozenset([RCC8Relation.TPP]))

    def test_inconsistent_constraint(self):
        """Test detecting inconsistent constraints."""
        network = ConstraintNetwork()
        network.add_variables("A", "B")

        network.add_constraint("A", "B", relation_set(RCC8Relation.DC))
        # Try to add contradictory constraint
        success = network.add_constraint("A", "B",
                                        relation_set(RCC8Relation.NTPP))
        self.assertFalse(success)

    def test_canonical_ordering(self):
        """Test that constraints use canonical variable ordering."""
        network = ConstraintNetwork()
        network.add_variables("A", "B")

        # Add constraint A-B
        network.add_constraint("A", "B", relation_set(RCC8Relation.TPP))

        # Query B-A should return inverse
        result = network.get_constraint("B", "A")
        self.assertEqual(result, frozenset([RCC8Relation.TPPi]))


class TestTractableFragments(unittest.TestCase):
    """Test tractable fragment checking."""

    def test_singletons_tractable(self):
        """Test that singleton relations are tractable."""
        for rel in RCC8Relation:
            is_tractable = TractableFragments.is_in_tractable_fragment(
                frozenset([rel])
            )
            self.assertTrue(is_tractable)

    def test_universal_tractable(self):
        """Test that universal relation is tractable."""
        is_tractable = TractableFragments.is_in_tractable_fragment(UNIVERSAL)
        self.assertTrue(is_tractable)


if __name__ == '__main__':
    unittest.main()
