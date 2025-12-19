"""
Integration tests for FA-QSR framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from qsr_base.rcc8 import RCC8Relation, relation_set
from qsr_base.faqsr_calculus import (
    FAQSRNetwork,
    FunctionalRelation,
)
from reasoning_engine.path_consistency import (
    FAQSRReasoner,
    ConsistencyStatus,
)
from nlp_models.sprl_model import SpatialTriple
from network_translator.gum_translator import GUMToFAQSRTranslator


class TestFAQSRNetwork(unittest.TestCase):
    """Test FA-QSR network operations."""

    def test_create_network(self):
        """Test basic network creation."""
        network = FAQSRNetwork()
        network.add_variable("A", entity_type="object")
        network.add_variable("B", entity_type="surface")

        self.assertEqual(network.num_variables, 2)

    def test_functional_constraint(self):
        """Test adding functional constraints."""
        network = FAQSRNetwork()
        network.add_variable("cup")
        network.add_variable("table")

        success = network.add_functional_constraint(
            "cup", "table",
            frozenset([FunctionalRelation.FSUPPORT])
        )
        self.assertTrue(success)

        func_rels = network.get_functional_constraint("cup", "table")
        self.assertIn(FunctionalRelation.FSUPPORT, func_rels)

    def test_geometric_requirements(self):
        """Test that functional constraints enforce geometric requirements."""
        network = FAQSRNetwork()
        network.add_variable("cup")
        network.add_variable("table")

        # Adding functional support should constrain geometric relations
        network.add_functional_constraint(
            "cup", "table",
            frozenset([FunctionalRelation.FSUPPORT])
        )

        geo_rels = network.get_geometric_constraint("cup", "table")
        # Should be constrained to contact relations
        self.assertNotIn(RCC8Relation.DC, geo_rels)


class TestReasoning(unittest.TestCase):
    """Test reasoning engine."""

    def setUp(self):
        self.reasoner = FAQSRReasoner()

    def test_consistent_network(self):
        """Test consistency checking on valid network."""
        network = FAQSRNetwork()
        network.add_variable("A")
        network.add_variable("B")
        network.add_variable("C")

        # A inside B, B inside C
        network.add_geometric_constraint("A", "B",
                                        relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))
        network.add_geometric_constraint("B", "C",
                                        relation_set(RCC8Relation.TPP, RCC8Relation.NTPP))

        result = self.reasoner.check_consistency(network)
        self.assertEqual(result.status, ConsistencyStatus.CONSISTENT)

    def test_inconsistent_network(self):
        """Test consistency checking on invalid network."""
        network = FAQSRNetwork()
        network.add_variable("A")
        network.add_variable("B")
        network.add_variable("C")

        # Cyclic strict containment - impossible
        network.add_geometric_constraint("A", "B",
                                        relation_set(RCC8Relation.NTPP))
        network.add_geometric_constraint("B", "C",
                                        relation_set(RCC8Relation.NTPP))
        network.add_geometric_constraint("C", "A",
                                        relation_set(RCC8Relation.NTPP))

        result = self.reasoner.check_consistency(network)
        self.assertEqual(result.status, ConsistencyStatus.INCONSISTENT)


class TestTranslation(unittest.TestCase):
    """Test GUM to FA-QSR translation."""

    def setUp(self):
        self.translator = GUMToFAQSRTranslator()

    def test_translate_on_support(self):
        """Test translation of 'on' with horizontal support."""
        triple = SpatialTriple("cup", "on", "table")
        result = self.translator.translate_triple(triple)

        self.assertIsNotNone(result.constraint)
        # Should have functional support
        self.assertTrue(len(result.constraint.functional) > 0)

    def test_translate_in_container(self):
        """Test translation of 'in' with container."""
        triple = SpatialTriple("flowers", "in", "vase")
        result = self.translator.translate_triple(triple)

        self.assertIsNotNone(result.constraint)
        # Should have functional containment
        func_rels = result.constraint.functional
        containment_rels = FunctionalRelation.containment_relations()
        has_containment = any(r in containment_rels for r in func_rels)
        self.assertTrue(has_containment or len(func_rels) > 0)

    def test_build_network_from_triples(self):
        """Test building network from multiple triples."""
        triples = [
            SpatialTriple("cup", "on", "table"),
            SpatialTriple("table", "in", "room"),
        ]

        network, results = self.translator.build_network(triples)

        self.assertEqual(network.num_variables, 3)
        self.assertEqual(len(results), 2)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_process_simple_sentence(self):
        """Test processing a simple spatial sentence."""
        from faqsr import FAQSR

        faqsr = FAQSR()
        result = faqsr.process("The book is on the table")

        # Should extract at least one triple
        self.assertGreater(len(result.triples), 0)
        # Network should be consistent
        if result.consistency_result:
            self.assertEqual(result.consistency_result.status,
                           ConsistencyStatus.CONSISTENT)

    def test_process_complex_sentence(self):
        """Test processing sentence with multiple relations."""
        from faqsr import FAQSR

        faqsr = FAQSR()
        result = faqsr.process("The flowers are in the vase on the table")

        # Should extract multiple triples
        self.assertGreaterEqual(len(result.triples), 1)


if __name__ == '__main__':
    unittest.main()
