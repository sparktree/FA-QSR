"""
Spatial Feature Extraction for SpRL

This module extracts linguistic and semantic features for spatial
role labeling and preposition disambiguation. Features are designed
to capture both syntactic structure and semantic affordances relevant
to functional spatial reasoning.

Feature categories:
1. Lexical features - Word forms, lemmas, POS tags
2. Dependency features - Syntactic relations, dependency paths
3. Context features - Surrounding words, sentence position
4. Semantic features - WordNet hypernyms, affordance cues

References:
- Kordjamshidi et al. (2011) - Spatial Role Labeling
- Pustejovsky et al. (2015) - ISO-Space
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re


class SpatialRole(Enum):
    """Spatial semantic roles following SpRL/ISO-Space annotation."""
    TRAJECTOR = "trajector"      # Figure - located entity
    LANDMARK = "landmark"        # Ground - reference entity
    INDICATOR = "indicator"      # Spatial trigger (preposition, verb)
    PATH = "path"               # Motion path
    DIRECTION = "direction"     # Directional modifier
    DISTANCE = "distance"       # Distance expression
    NONE = "none"              # Not a spatial role


class PrepositionSense(Enum):
    """Semantic senses for spatial prepositions."""
    # Support senses
    ON_SUPPORT = "on_support"           # Horizontal surface support
    ON_VERTICAL = "on_vertical"         # Vertical attachment (wall)
    ON_HANGING = "on_hanging"           # Hanging support (hook)

    # Containment senses
    IN_CONTAINER = "in_container"       # Physical containment
    IN_REGION = "in_region"             # Regional location
    IN_FUNCTIONAL = "in_functional"     # Functional enclosure

    # Proximity senses
    AT_LOCATION = "at_location"         # Proximal location
    NEAR_PROXIMITY = "near_proximity"   # Close proximity
    BY_ADJACENCY = "by_adjacency"       # Adjacent position

    # Projective senses
    ABOVE_VERTICAL = "above_vertical"   # Vertical projection above
    BELOW_VERTICAL = "below_vertical"   # Vertical projection below
    FRONT_PROJECTIVE = "front_projective"  # Front projection
    BEHIND_PROJECTIVE = "behind_projective"  # Back projection

    # Other
    GENERIC = "generic"                 # Underspecified
    UNKNOWN = "unknown"                 # Cannot determine


@dataclass
class Token:
    """
    A token with linguistic annotations.

    Represents a word with its associated linguistic features
    as would be provided by a dependency parser.
    """
    text: str
    lemma: str
    pos: str  # Part of speech (Penn Treebank style)
    dep: str  # Dependency relation
    head_idx: int  # Index of head token
    idx: int  # Token index in sentence
    ner: str = "O"  # Named entity tag
    is_spatial_trigger: bool = False

    def __repr__(self) -> str:
        return f"Token({self.text}/{self.pos})"


@dataclass
class DependencyFeatures:
    """Features derived from dependency structure."""
    # Direct dependency relations
    dep_to_head: str              # Dependency relation to head
    head_pos: str                 # POS of head
    head_lemma: str              # Lemma of head

    # Children features
    num_children: int            # Number of dependents
    child_deps: List[str]        # Dependency relations of children
    child_pos_tags: List[str]    # POS tags of children

    # Path features
    path_to_root: List[str]      # Dependency path to root
    path_length: int             # Distance to root

    # Specific patterns
    has_prep_child: bool         # Has prepositional phrase child
    has_nominal_child: bool      # Has noun phrase child
    is_prep_object: bool         # Is object of preposition


@dataclass
class LexicalFeatures:
    """Lexical and morphological features."""
    # Word forms
    word: str
    lemma: str
    pos_tag: str
    pos_fine: str  # Fine-grained POS if available

    # Morphological
    is_capitalized: bool
    is_all_caps: bool
    has_digit: bool
    word_length: int

    # Lexical class membership
    is_spatial_prep: bool        # In spatial preposition list
    is_motion_verb: bool         # In motion verb list
    is_locative_noun: bool       # In locative noun list

    # N-gram features
    prefix_2: str               # First 2 characters
    prefix_3: str               # First 3 characters
    suffix_2: str               # Last 2 characters
    suffix_3: str               # Last 3 characters


@dataclass
class ContextFeatures:
    """Contextual features from surrounding words."""
    # Window features
    prev_word: Optional[str]
    prev_pos: Optional[str]
    prev_2_word: Optional[str]
    prev_2_pos: Optional[str]
    next_word: Optional[str]
    next_pos: Optional[str]
    next_2_word: Optional[str]
    next_2_pos: Optional[str]

    # Position features
    sentence_position: float     # Relative position (0-1)
    distance_from_start: int
    distance_from_end: int

    # Bag-of-words in window
    words_in_window: Set[str]
    pos_in_window: Set[str]


@dataclass
class SemanticFeatures:
    """Semantic and conceptual features."""
    # Affordance cues
    affords_support: bool       # Can serve as support surface
    affords_containment: bool   # Can serve as container
    is_animate: bool           # Is animate entity
    is_concrete: bool          # Is concrete (vs abstract)

    # Conceptual type
    object_type: str           # container, surface, region, etc.
    typical_function: str      # Primary function if known

    # WordNet-derived (if available)
    hypernym_chain: List[str]  # Hypernyms up to entity
    is_artifact: bool
    is_location: bool


@dataclass
class FeatureVector:
    """
    Complete feature vector for a token or span.

    Combines all feature categories into a single representation
    suitable for machine learning models.
    """
    lexical: LexicalFeatures
    dependency: Optional[DependencyFeatures] = None
    context: Optional[ContextFeatures] = None
    semantic: Optional[SemanticFeatures] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for ML models."""
        features = {}

        # Lexical features
        features['word'] = self.lexical.word
        features['lemma'] = self.lexical.lemma
        features['pos'] = self.lexical.pos_tag
        features['is_spatial_prep'] = self.lexical.is_spatial_prep
        features['is_motion_verb'] = self.lexical.is_motion_verb
        features['word_length'] = self.lexical.word_length

        # Dependency features
        if self.dependency:
            features['dep_rel'] = self.dependency.dep_to_head
            features['head_pos'] = self.dependency.head_pos
            features['path_length'] = self.dependency.path_length
            features['is_prep_object'] = self.dependency.is_prep_object

        # Context features
        if self.context:
            features['prev_pos'] = self.context.prev_pos or 'NONE'
            features['next_pos'] = self.context.next_pos or 'NONE'
            features['sent_position'] = self.context.sentence_position

        # Semantic features
        if self.semantic:
            features['affords_support'] = self.semantic.affords_support
            features['affords_containment'] = self.semantic.affords_containment
            features['object_type'] = self.semantic.object_type

        return features


class SpatialFeatureExtractor:
    """
    Feature extractor for spatial role labeling.

    Extracts linguistic features from tokenized and parsed sentences
    to support spatial role labeling and preposition disambiguation.
    """

    # Spatial prepositions
    SPATIAL_PREPOSITIONS = {
        # Topological/Functional
        'in', 'inside', 'within', 'into',
        'on', 'onto', 'upon', 'atop',
        'at', 'by', 'beside', 'alongside',
        'near', 'close', 'next',
        'under', 'beneath', 'below', 'underneath',
        'over', 'above',
        'behind', 'after',
        'before', 'ahead', 'in front of',
        'between', 'among', 'amongst',
        'through', 'across', 'along',
        'around', 'about',
        'against', 'toward', 'towards',
        'from', 'off', 'out', 'away',
        # Compound (common)
        'outside', 'inside', 'alongside',
    }

    # Motion verbs
    MOTION_VERBS = {
        'go', 'come', 'move', 'travel', 'walk', 'run',
        'enter', 'exit', 'leave', 'arrive', 'depart',
        'put', 'place', 'set', 'lay', 'position',
        'push', 'pull', 'slide', 'roll',
        'rise', 'fall', 'climb', 'descend',
        'cross', 'pass', 'reach', 'approach',
    }

    # Locative nouns (common spatial reference entities)
    LOCATIVE_NOUNS = {
        'room', 'house', 'building', 'office', 'store',
        'table', 'desk', 'shelf', 'floor', 'wall', 'ceiling',
        'box', 'container', 'bag', 'basket', 'drawer',
        'street', 'road', 'path', 'corner', 'intersection',
        'city', 'town', 'country', 'region', 'area',
        'top', 'bottom', 'side', 'front', 'back', 'center', 'edge',
    }

    # Object affordances (simplified knowledge base)
    AFFORDANCES = {
        # Containers
        'box': {'containment': True, 'support': False},
        'bag': {'containment': True, 'support': False},
        'cup': {'containment': True, 'support': False},
        'bowl': {'containment': True, 'support': False},
        'vase': {'containment': True, 'support': True},
        'basket': {'containment': True, 'support': False},
        'drawer': {'containment': True, 'support': True},
        'room': {'containment': True, 'support': False},

        # Support surfaces
        'table': {'containment': False, 'support': True},
        'desk': {'containment': False, 'support': True},
        'shelf': {'containment': False, 'support': True},
        'floor': {'containment': False, 'support': True},
        'ground': {'containment': False, 'support': True},
        'counter': {'containment': False, 'support': True},

        # Vertical supports
        'wall': {'containment': False, 'support': True, 'orientation': 'vertical'},
        'hook': {'containment': False, 'support': True, 'type': 'hanging'},
        'peg': {'containment': False, 'support': True, 'type': 'hanging'},
        'nail': {'containment': False, 'support': True, 'type': 'hanging'},

        # Regions
        'city': {'containment': True, 'support': False, 'type': 'region'},
        'country': {'containment': True, 'support': False, 'type': 'region'},
        'area': {'containment': True, 'support': False, 'type': 'region'},
    }

    def __init__(self, use_semantic_features: bool = True):
        """
        Initialize the feature extractor.

        Args:
            use_semantic_features: Whether to include semantic/affordance features
        """
        self.use_semantic_features = use_semantic_features

    def extract_features(self, tokens: List[Token], target_idx: int) -> FeatureVector:
        """
        Extract features for a specific token.

        Args:
            tokens: List of tokens in the sentence
            target_idx: Index of the token to extract features for

        Returns:
            Complete feature vector for the token
        """
        token = tokens[target_idx]

        # Extract each feature category
        lexical = self._extract_lexical(token)
        dependency = self._extract_dependency(tokens, target_idx)
        context = self._extract_context(tokens, target_idx)

        semantic = None
        if self.use_semantic_features:
            semantic = self._extract_semantic(token)

        return FeatureVector(
            lexical=lexical,
            dependency=dependency,
            context=context,
            semantic=semantic
        )

    def extract_pair_features(self, tokens: List[Token],
                             trajector_idx: int,
                             landmark_idx: int,
                             indicator_idx: int) -> Dict[str, Any]:
        """
        Extract features for a trajector-indicator-landmark triple.

        Args:
            tokens: Sentence tokens
            trajector_idx: Index of trajector (figure)
            landmark_idx: Index of landmark (ground)
            indicator_idx: Index of spatial indicator (preposition)

        Returns:
            Dictionary of pair-wise features
        """
        features = {}

        # Individual features
        traj_feats = self.extract_features(tokens, trajector_idx)
        land_feats = self.extract_features(tokens, landmark_idx)
        ind_feats = self.extract_features(tokens, indicator_idx)

        features['trajector_lemma'] = traj_feats.lexical.lemma
        features['landmark_lemma'] = land_feats.lexical.lemma
        features['indicator_lemma'] = ind_feats.lexical.lemma

        features['trajector_pos'] = traj_feats.lexical.pos_tag
        features['landmark_pos'] = land_feats.lexical.pos_tag
        features['indicator_pos'] = ind_feats.lexical.pos_tag

        # Distance features
        features['traj_ind_distance'] = abs(trajector_idx - indicator_idx)
        features['land_ind_distance'] = abs(landmark_idx - indicator_idx)
        features['traj_land_distance'] = abs(trajector_idx - landmark_idx)

        # Order features
        features['traj_before_ind'] = trajector_idx < indicator_idx
        features['land_after_ind'] = landmark_idx > indicator_idx

        # Semantic features
        if self.use_semantic_features:
            traj_aff = self.AFFORDANCES.get(traj_feats.lexical.lemma.lower(), {})
            land_aff = self.AFFORDANCES.get(land_feats.lexical.lemma.lower(), {})

            features['landmark_affords_support'] = land_aff.get('support', False)
            features['landmark_affords_containment'] = land_aff.get('containment', False)
            features['landmark_orientation'] = land_aff.get('orientation', 'horizontal')
            features['landmark_type'] = land_aff.get('type', 'object')

        return features

    def _extract_lexical(self, token: Token) -> LexicalFeatures:
        """Extract lexical features for a token."""
        word = token.text
        lemma = token.lemma.lower()

        return LexicalFeatures(
            word=word,
            lemma=lemma,
            pos_tag=token.pos,
            pos_fine=token.pos,  # Would be more specific with full tagger
            is_capitalized=word[0].isupper() if word else False,
            is_all_caps=word.isupper() if word else False,
            has_digit=bool(re.search(r'\d', word)),
            word_length=len(word),
            is_spatial_prep=lemma in self.SPATIAL_PREPOSITIONS,
            is_motion_verb=lemma in self.MOTION_VERBS,
            is_locative_noun=lemma in self.LOCATIVE_NOUNS,
            prefix_2=word[:2] if len(word) >= 2 else word,
            prefix_3=word[:3] if len(word) >= 3 else word,
            suffix_2=word[-2:] if len(word) >= 2 else word,
            suffix_3=word[-3:] if len(word) >= 3 else word,
        )

    def _extract_dependency(self, tokens: List[Token], idx: int) -> DependencyFeatures:
        """Extract dependency-based features."""
        token = tokens[idx]
        head_idx = token.head_idx

        # Get head features
        if 0 <= head_idx < len(tokens) and head_idx != idx:
            head = tokens[head_idx]
            head_pos = head.pos
            head_lemma = head.lemma
        else:
            head_pos = "ROOT"
            head_lemma = "ROOT"

        # Get children
        children = [t for i, t in enumerate(tokens) if t.head_idx == idx and i != idx]
        child_deps = [c.dep for c in children]
        child_pos = [c.pos for c in children]

        # Path to root
        path = []
        current_idx = idx
        visited = {idx}
        while current_idx >= 0 and current_idx < len(tokens):
            current = tokens[current_idx]
            if current.head_idx in visited or current.head_idx == current_idx:
                break
            path.append(current.dep)
            visited.add(current.head_idx)
            current_idx = current.head_idx

        return DependencyFeatures(
            dep_to_head=token.dep,
            head_pos=head_pos,
            head_lemma=head_lemma,
            num_children=len(children),
            child_deps=child_deps,
            child_pos_tags=child_pos,
            path_to_root=path,
            path_length=len(path),
            has_prep_child=any(t.pos == 'IN' for t in children),
            has_nominal_child=any(t.pos.startswith('NN') for t in children),
            is_prep_object=token.dep == 'pobj',
        )

    def _extract_context(self, tokens: List[Token], idx: int) -> ContextFeatures:
        """Extract context window features."""
        n = len(tokens)

        prev_word = tokens[idx - 1].text if idx > 0 else None
        prev_pos = tokens[idx - 1].pos if idx > 0 else None
        prev_2_word = tokens[idx - 2].text if idx > 1 else None
        prev_2_pos = tokens[idx - 2].pos if idx > 1 else None

        next_word = tokens[idx + 1].text if idx < n - 1 else None
        next_pos = tokens[idx + 1].pos if idx < n - 1 else None
        next_2_word = tokens[idx + 2].text if idx < n - 2 else None
        next_2_pos = tokens[idx + 2].pos if idx < n - 2 else None

        # Window words and POS
        window_start = max(0, idx - 3)
        window_end = min(n, idx + 4)
        words_in_window = {tokens[i].text.lower() for i in range(window_start, window_end)}
        pos_in_window = {tokens[i].pos for i in range(window_start, window_end)}

        return ContextFeatures(
            prev_word=prev_word,
            prev_pos=prev_pos,
            prev_2_word=prev_2_word,
            prev_2_pos=prev_2_pos,
            next_word=next_word,
            next_pos=next_pos,
            next_2_word=next_2_word,
            next_2_pos=next_2_pos,
            sentence_position=idx / n if n > 0 else 0.0,
            distance_from_start=idx,
            distance_from_end=n - idx - 1,
            words_in_window=words_in_window,
            pos_in_window=pos_in_window,
        )

    def _extract_semantic(self, token: Token) -> SemanticFeatures:
        """Extract semantic/affordance features."""
        lemma = token.lemma.lower()
        affordances = self.AFFORDANCES.get(lemma, {})

        return SemanticFeatures(
            affords_support=affordances.get('support', False),
            affords_containment=affordances.get('containment', False),
            is_animate=False,  # Would need NER or WordNet
            is_concrete=True,   # Default assumption
            object_type=affordances.get('type', 'object'),
            typical_function='unknown',
            hypernym_chain=[],  # Would need WordNet
            is_artifact=True,
            is_location=lemma in self.LOCATIVE_NOUNS,
        )

    def identify_spatial_indicators(self, tokens: List[Token]) -> List[int]:
        """
        Identify potential spatial indicators in a sentence.

        Returns indices of tokens that are likely spatial prepositions
        or other spatial triggers.
        """
        indicators = []
        for i, token in enumerate(tokens):
            if token.lemma.lower() in self.SPATIAL_PREPOSITIONS:
                indicators.append(i)
            elif token.lemma.lower() in self.MOTION_VERBS:
                indicators.append(i)
        return indicators


def demonstrate_features():
    """Demonstrate feature extraction."""
    print("Spatial Feature Extraction Demonstration")
    print("=" * 50)

    # Create sample sentence: "The flowers are in the vase on the table"
    tokens = [
        Token("The", "the", "DT", "det", 1, 0),
        Token("flowers", "flower", "NNS", "nsubj", 2, 1),
        Token("are", "be", "VBP", "ROOT", 2, 2),
        Token("in", "in", "IN", "prep", 2, 3, is_spatial_trigger=True),
        Token("the", "the", "DT", "det", 5, 4),
        Token("vase", "vase", "NN", "pobj", 3, 5),
        Token("on", "on", "IN", "prep", 5, 6, is_spatial_trigger=True),
        Token("the", "the", "DT", "det", 8, 7),
        Token("table", "table", "NN", "pobj", 6, 8),
    ]

    extractor = SpatialFeatureExtractor()

    print("\nSentence: 'The flowers are in the vase on the table'")
    print("-" * 50)

    # Extract features for "in"
    print("\nFeatures for 'in' (indicator):")
    in_features = extractor.extract_features(tokens, 3)
    print(f"  Lexical: {in_features.lexical.lemma}, is_spatial_prep={in_features.lexical.is_spatial_prep}")
    print(f"  Dependency: dep={in_features.dependency.dep_to_head}, head_pos={in_features.dependency.head_pos}")
    print(f"  Context: prev={in_features.context.prev_word}, next={in_features.context.next_word}")

    # Extract features for "vase"
    print("\nFeatures for 'vase' (landmark):")
    vase_features = extractor.extract_features(tokens, 5)
    print(f"  Lexical: {vase_features.lexical.lemma}, is_locative={vase_features.lexical.is_locative_noun}")
    print(f"  Semantic: affords_containment={vase_features.semantic.affords_containment}")

    # Extract pair features
    print("\nPair features for (flowers, in, vase):")
    pair_feats = extractor.extract_pair_features(tokens, 1, 5, 3)
    for key, value in list(pair_feats.items())[:8]:
        print(f"  {key}: {value}")

    # Identify spatial indicators
    print("\nIdentified spatial indicators:")
    indicators = extractor.identify_spatial_indicators(tokens)
    for idx in indicators:
        print(f"  {idx}: '{tokens[idx].text}'")


if __name__ == "__main__":
    demonstrate_features()
