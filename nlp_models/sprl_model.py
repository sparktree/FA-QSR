"""
Spatial Role Labeling (SpRL) Model

This module implements a Spatial Role Labeler that identifies:
- Trajectors (figures): entities being located
- Landmarks (grounds): reference entities
- Spatial Indicators: prepositions/verbs expressing spatial relations

The model can be trained on annotated corpora such as:
- SpaceEval (ISO-Space standard)
- CLEF IAPR TC-12 SpRL dataset
- SemEval spatial relations tasks

References:
- Kordjamshidi et al. (2011) - Spatial Role Labeling: Task Definition and Annotation Scheme
- Pustejovsky et al. (2015) - ISO-Space: The Annotation of Spatial Information in Language
"""

from typing import List, Dict, Optional, Tuple, Any, Sequence
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.spatial_features import (
    Token,
    SpatialRole,
    SpatialFeatureExtractor,
    FeatureVector,
)


@dataclass
class SpRLAnnotation:
    """
    Annotation for a single spatial expression.

    Follows ISO-Space / SpRL annotation standards.
    """
    # Core roles
    trajector: Optional[Tuple[int, int]] = None  # (start_idx, end_idx)
    landmark: Optional[Tuple[int, int]] = None
    indicator: Optional[Tuple[int, int]] = None

    # Text spans
    trajector_text: str = ""
    landmark_text: str = ""
    indicator_text: str = ""

    # Additional spatial elements
    path: Optional[Tuple[int, int]] = None
    direction: Optional[str] = None
    distance: Optional[str] = None

    # Semantic type
    spatial_type: str = "static"  # static, dynamic, motion
    relation_type: str = "topological"  # topological, directional, distance

    # Confidence score (0-1)
    confidence: float = 1.0


@dataclass
class SpatialTriple:
    """
    A spatial relation triple: <trajector, indicator, landmark>

    This is the primary output format for FA-QSR constraint generation.
    """
    trajector: str           # Trajector entity/span
    indicator: str           # Spatial indicator (preposition)
    landmark: str            # Landmark entity/span

    # Metadata
    trajector_type: str = "entity"
    landmark_type: str = "entity"
    indicator_sense: str = "generic"

    # Indices in source sentence
    trajector_idx: Optional[int] = None
    landmark_idx: Optional[int] = None
    indicator_idx: Optional[int] = None

    def __repr__(self) -> str:
        return f"<{self.trajector}, {self.indicator}, {self.landmark}>"


class LabelingStrategy(Enum):
    """Labeling strategies for SpRL."""
    BIO = "bio"              # Begin-Inside-Outside
    BILOU = "bilou"          # Begin-Inside-Last-Outside-Unit
    IO = "io"                # Inside-Outside only


class SpatialRoleLabeler:
    """
    Spatial Role Labeling model.

    Identifies trajectors, landmarks, and spatial indicators in text.
    Uses a feature-based approach suitable for CRF or neural models.
    """

    def __init__(self,
                 labeling_strategy: LabelingStrategy = LabelingStrategy.BIO,
                 use_heuristics: bool = True):
        """
        Initialize the SpRL model.

        Args:
            labeling_strategy: BIO tagging scheme to use
            use_heuristics: Whether to use rule-based heuristics alongside ML
        """
        self.labeling_strategy = labeling_strategy
        self.use_heuristics = use_heuristics
        self.feature_extractor = SpatialFeatureExtractor()

        # Model state (would be loaded from trained model)
        self._indicator_model = None
        self._trajector_model = None
        self._landmark_model = None
        self._is_trained = False

    def label(self, tokens: List[Token]) -> List[SpRLAnnotation]:
        """
        Label spatial roles in a tokenized sentence.

        Args:
            tokens: List of tokens with POS and dependency annotations

        Returns:
            List of spatial annotations found
        """
        if not tokens:
            return []

        # Step 1: Identify spatial indicators
        indicators = self._identify_indicators(tokens)

        if not indicators:
            return []

        # Step 2: For each indicator, find trajector and landmark
        annotations = []
        for ind_idx in indicators:
            annotation = self._find_roles_for_indicator(tokens, ind_idx)
            if annotation:
                annotations.append(annotation)

        return annotations

    def extract_triples(self, tokens: List[Token]) -> List[SpatialTriple]:
        """
        Extract spatial triples from tokenized text.

        This is the primary interface for FA-QSR integration.

        Args:
            tokens: Tokenized sentence

        Returns:
            List of spatial triples
        """
        annotations = self.label(tokens)
        triples = []

        for ann in annotations:
            if ann.trajector and ann.landmark and ann.indicator:
                traj_text = ann.trajector_text or self._get_span_text(tokens, ann.trajector)
                land_text = ann.landmark_text or self._get_span_text(tokens, ann.landmark)
                ind_text = ann.indicator_text or self._get_span_text(tokens, ann.indicator)

                triple = SpatialTriple(
                    trajector=traj_text,
                    indicator=ind_text,
                    landmark=land_text,
                    trajector_idx=ann.trajector[0],
                    landmark_idx=ann.landmark[0],
                    indicator_idx=ann.indicator[0],
                )
                triples.append(triple)

        return triples

    def _identify_indicators(self, tokens: List[Token]) -> List[int]:
        """Identify spatial indicators using heuristics and/or model."""
        if self.use_heuristics:
            return self._heuristic_indicator_detection(tokens)
        elif self._is_trained:
            return self._model_indicator_detection(tokens)
        else:
            return self._heuristic_indicator_detection(tokens)

    def _heuristic_indicator_detection(self, tokens: List[Token]) -> List[int]:
        """Rule-based spatial indicator detection."""
        indicators = []

        for i, token in enumerate(tokens):
            lemma = token.lemma.lower()

            # Check preposition list
            if lemma in self.feature_extractor.SPATIAL_PREPOSITIONS:
                # Verify it's functioning as a preposition
                if token.pos in ('IN', 'TO', 'RP'):
                    indicators.append(i)

            # Check for motion verbs with spatial arguments
            elif lemma in self.feature_extractor.MOTION_VERBS:
                # Check if followed by spatial preposition
                if i + 1 < len(tokens):
                    next_lemma = tokens[i + 1].lemma.lower()
                    if next_lemma in self.feature_extractor.SPATIAL_PREPOSITIONS:
                        indicators.append(i)

        return indicators

    def _model_indicator_detection(self, tokens: List[Token]) -> List[int]:
        """Model-based spatial indicator detection."""
        # Placeholder - would use trained sequence labeling model
        return self._heuristic_indicator_detection(tokens)

    def _find_roles_for_indicator(self, tokens: List[Token],
                                   ind_idx: int) -> Optional[SpRLAnnotation]:
        """
        Find trajector and landmark for a given spatial indicator.

        Uses dependency structure and heuristics to identify arguments.
        """
        indicator = tokens[ind_idx]

        # Find trajector (typically the subject or preceding noun)
        trajector_idx = self._find_trajector(tokens, ind_idx)

        # Find landmark (typically the object of the preposition)
        landmark_idx = self._find_landmark(tokens, ind_idx)

        if trajector_idx is None or landmark_idx is None:
            return None

        return SpRLAnnotation(
            trajector=(trajector_idx, trajector_idx + 1),
            landmark=(landmark_idx, landmark_idx + 1),
            indicator=(ind_idx, ind_idx + 1),
            trajector_text=tokens[trajector_idx].text,
            landmark_text=tokens[landmark_idx].text,
            indicator_text=indicator.text,
        )

    def _find_trajector(self, tokens: List[Token], ind_idx: int) -> Optional[int]:
        """
        Find the trajector (figure) for a spatial indicator.

        Heuristics:
        1. Subject of the clause containing the preposition
        2. Noun phrase immediately preceding the preposition
        3. Head of the prepositional phrase's governor
        """
        indicator = tokens[ind_idx]

        # Strategy 1: Look for subject in dependency structure
        # Find the head of the preposition's phrase
        current_idx = indicator.head_idx
        while current_idx >= 0 and current_idx < len(tokens):
            current = tokens[current_idx]

            # Look for subject among current's children
            for i, t in enumerate(tokens):
                if t.head_idx == current_idx and t.dep in ('nsubj', 'nsubjpass'):
                    return i

            if current.head_idx == current_idx:
                break
            current_idx = current.head_idx

        # Strategy 2: Nearest preceding noun
        for i in range(ind_idx - 1, -1, -1):
            if tokens[i].pos.startswith('NN'):
                return i

        return None

    def _find_landmark(self, tokens: List[Token], ind_idx: int) -> Optional[int]:
        """
        Find the landmark (ground) for a spatial indicator.

        Heuristics:
        1. Direct object of the preposition (pobj)
        2. Noun phrase following the preposition
        """
        # Strategy 1: Look for pobj dependent of preposition
        for i, t in enumerate(tokens):
            if t.head_idx == ind_idx and t.dep == 'pobj':
                return i

        # Strategy 2: Next noun after preposition
        for i in range(ind_idx + 1, len(tokens)):
            if tokens[i].pos.startswith('NN'):
                return i

        return None

    def _get_span_text(self, tokens: List[Token], span: Tuple[int, int]) -> str:
        """Get text for a token span."""
        return ' '.join(tokens[i].text for i in range(span[0], span[1]))

    def train(self, training_data: List[Tuple[List[Token], List[SpRLAnnotation]]]):
        """
        Train the SpRL model on annotated data.

        Args:
            training_data: List of (tokens, annotations) pairs
        """
        # Extract features for all training instances
        X_indicator = []
        y_indicator = []
        X_roles = []
        y_roles = []

        for tokens, annotations in training_data:
            # Create token-level labels
            indicator_labels = ['O'] * len(tokens)
            trajector_labels = ['O'] * len(tokens)
            landmark_labels = ['O'] * len(tokens)

            for ann in annotations:
                if ann.indicator:
                    self._mark_span(indicator_labels, ann.indicator, 'IND')
                if ann.trajector:
                    self._mark_span(trajector_labels, ann.trajector, 'TRAJ')
                if ann.landmark:
                    self._mark_span(landmark_labels, ann.landmark, 'LAND')

            # Extract features
            for i, token in enumerate(tokens):
                features = self.feature_extractor.extract_features(tokens, i)
                X_indicator.append(features.to_dict())
                y_indicator.append(indicator_labels[i])

        # Would train actual ML model here (CRF, BiLSTM-CRF, etc.)
        self._is_trained = True
        print(f"Training complete with {len(training_data)} examples")

    def _mark_span(self, labels: List[str], span: Tuple[int, int], tag: str):
        """Mark a span with BIO tags."""
        start, end = span
        if self.labeling_strategy == LabelingStrategy.BIO:
            labels[start] = f'B-{tag}'
            for i in range(start + 1, end):
                labels[i] = f'I-{tag}'

    def evaluate(self, test_data: List[Tuple[List[Token], List[SpRLAnnotation]]]) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Returns precision, recall, F1 for each role type.
        """
        metrics = {
            'indicator_precision': 0.0,
            'indicator_recall': 0.0,
            'indicator_f1': 0.0,
            'trajector_precision': 0.0,
            'trajector_recall': 0.0,
            'trajector_f1': 0.0,
            'landmark_precision': 0.0,
            'landmark_recall': 0.0,
            'landmark_f1': 0.0,
            'overall_f1': 0.0,
        }

        true_positives = {'ind': 0, 'traj': 0, 'land': 0}
        false_positives = {'ind': 0, 'traj': 0, 'land': 0}
        false_negatives = {'ind': 0, 'traj': 0, 'land': 0}

        for tokens, gold_annotations in test_data:
            pred_annotations = self.label(tokens)

            # Compare predictions to gold
            for gold in gold_annotations:
                gold_ind = gold.indicator
                gold_traj = gold.trajector
                gold_land = gold.landmark

                # Check if any prediction matches
                matched = False
                for pred in pred_annotations:
                    if self._spans_overlap(pred.indicator, gold_ind):
                        true_positives['ind'] += 1
                    if self._spans_overlap(pred.trajector, gold_traj):
                        true_positives['traj'] += 1
                    if self._spans_overlap(pred.landmark, gold_land):
                        true_positives['land'] += 1
                    matched = True

                if not matched:
                    if gold_ind:
                        false_negatives['ind'] += 1
                    if gold_traj:
                        false_negatives['traj'] += 1
                    if gold_land:
                        false_negatives['land'] += 1

        # Calculate metrics
        for role in ['ind', 'traj', 'land']:
            tp = true_positives[role]
            fp = false_positives[role]
            fn = false_negatives[role]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            role_name = {'ind': 'indicator', 'traj': 'trajector', 'land': 'landmark'}[role]
            metrics[f'{role_name}_precision'] = precision
            metrics[f'{role_name}_recall'] = recall
            metrics[f'{role_name}_f1'] = f1

        # Overall F1
        total_f1 = (metrics['indicator_f1'] + metrics['trajector_f1'] + metrics['landmark_f1']) / 3
        metrics['overall_f1'] = total_f1

        return metrics

    def _spans_overlap(self, span1: Optional[Tuple[int, int]],
                       span2: Optional[Tuple[int, int]]) -> bool:
        """Check if two spans overlap."""
        if span1 is None or span2 is None:
            return False
        return span1[0] < span2[1] and span2[0] < span1[1]


def demonstrate_sprl():
    """Demonstrate SpRL model capabilities."""
    print("Spatial Role Labeling Demonstration")
    print("=" * 50)

    # Sample sentences with dependency annotations
    sentences = [
        # "The book is on the table"
        [
            Token("The", "the", "DT", "det", 1, 0),
            Token("book", "book", "NN", "nsubj", 2, 1),
            Token("is", "be", "VBZ", "ROOT", 2, 2),
            Token("on", "on", "IN", "prep", 2, 3),
            Token("the", "the", "DT", "det", 5, 4),
            Token("table", "table", "NN", "pobj", 3, 5),
        ],
        # "The cat sleeps in the box under the bed"
        [
            Token("The", "the", "DT", "det", 1, 0),
            Token("cat", "cat", "NN", "nsubj", 2, 1),
            Token("sleeps", "sleep", "VBZ", "ROOT", 2, 2),
            Token("in", "in", "IN", "prep", 2, 3),
            Token("the", "the", "DT", "det", 5, 4),
            Token("box", "box", "NN", "pobj", 3, 5),
            Token("under", "under", "IN", "prep", 5, 6),
            Token("the", "the", "DT", "det", 8, 7),
            Token("bed", "bed", "NN", "pobj", 6, 8),
        ],
    ]

    labeler = SpatialRoleLabeler()

    for i, tokens in enumerate(sentences):
        sentence = ' '.join(t.text for t in tokens)
        print(f"\n{i+1}. {sentence}")
        print("-" * 40)

        # Get annotations
        annotations = labeler.label(tokens)
        print(f"Found {len(annotations)} spatial expression(s)")

        for ann in annotations:
            print(f"  Trajector: {ann.trajector_text}")
            print(f"  Indicator: {ann.indicator_text}")
            print(f"  Landmark: {ann.landmark_text}")
            print()

        # Get triples
        triples = labeler.extract_triples(tokens)
        print("Spatial Triples:")
        for triple in triples:
            print(f"  {triple}")


if __name__ == "__main__":
    demonstrate_sprl()
