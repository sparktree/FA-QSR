# FA-QSR: Functionally-Augmented Qualitative Spatial Reasoning
A synthesis of formal semantic ontology and qualitative spatial reasoning.
A Python framework that extends standard mereotopological calculi with first-class functional relations, enabling computational reasoning over both geometric and functional dimensions of spatial language.

## Overview

When a speaker says "the book is on the table," they convey more than geometric adjacency—they implicate that the table *supports* the book, that gravity acts upon this configuration, and that removing the table would cause the book to fall. FA-QSR captures both the *where* (geometric configuration) and the *what* (functional relationships) of spatial meaning.

### Key Features

- **Hybrid Spatial Representation**: Combines RCC-8 geometric relations with functional primitives (`fsupport`, `fcontainment`)
- **Natural Language Processing**: Extracts spatial triples from text using Spatial Role Labeling (SpRL)
- **Preposition Sense Disambiguation**: Distinguishes between functional and geometric readings (e.g., "on" as support vs. "above" as projection)
- **Constraint-Based Reasoning**: Path consistency and backtracking algorithms for spatial inference
- **Complexity Analysis**: Classifies constraint fragments as tractable (polynomial) or NP-hard
- **GUM Ontology Bridge**: Maps linguistic categories to formal spatial calculi

## Installation

### Requirements

- Python 3.7+
- No external dependencies required for core functionality

### Setup

```bash
git clone https://github.com/yourusername/FAQSR.git
cd FAQSR
```

### Optional Dependencies

For enhanced functionality, install optional packages:

```bash
# NLP (production-quality parsing)
pip install spacy>=3.0.0
python -m spacy download en_core_web_sm

# Ontology processing
pip install owlready2>=0.35 rdflib>=6.0.0

# Testing
pip install pytest>=7.0.0 pytest-cov>=3.0.0
```

## Quick Start

### Basic Usage

```python
from faqsr import FAQSR

# Initialize the framework
faqsr = FAQSR()

# Process natural language spatial text
result = faqsr.process("The flowers are in the vase on the table")

# View extracted spatial triples
for triple in result.triples:
    print(f"{triple.trajector} --[{triple.indicator}]--> {triple.landmark}")

# Check consistency
print(f"Consistent: {result.consistency_result.status.value}")

# View complexity classification
print(f"Complexity: {result.complexity_result.complexity_class.value}")
```

### Direct Network Construction

```python
from qsr_base.faqsr_calculus import FAQSRNetwork, FunctionalRelation
from qsr_base.rcc8 import RCC8Relation, relation_set

# Create a network manually
network = FAQSRNetwork()

# Add spatial entities
network.add_variable("cup", entity_type="object")
network.add_variable("table", entity_type="surface", affordances={"support"})
network.add_variable("room", entity_type="region")

# Add functional constraint (cup supported by table)
network.add_functional_constraint(
    "cup", "table",
    frozenset([FunctionalRelation.FSUPPORT])
)

# Add geometric constraint (table inside room)
network.add_geometric_constraint(
    "table", "room",
    relation_set(RCC8Relation.TPP, RCC8Relation.NTPP)
)

# Perform inference
inference = faqsr.infer(network, "cup", "room")
print(f"Inferred geometric relations: {inference.geometric_relations}")
```

### Processing Pre-extracted Triples

```python
from nlp_models.sprl_model import SpatialTriple

triples = [
    SpatialTriple("cup", "on", "saucer"),
    SpatialTriple("saucer", "on", "table"),
    SpatialTriple("table", "in", "kitchen"),
]

result = faqsr.process_triples(triples)
print(f"Network has {result.network.num_variables} entities")
```

## Project Structure

```
FAQSR/
├── faqsr.py                    # Main orchestration module
├── config.py                   # Configuration and constants
├── requirements.txt            # Dependencies
│
├── qsr_base/                   # Core spatial calculi
│   ├── rcc8.py                 # RCC-8 implementation (8 JEPD relations, composition table)
│   └── faqsr_calculus.py       # FA-QSR extension with functional primitives
│
├── reasoning_engine/           # Inference algorithms
│   ├── path_consistency.py     # Path consistency (polynomial for tractable fragments)
│   └── backtrack.py            # Backtracking search (for NP-hard fragments)
│
├── nlp_models/                 # Natural language processing
│   ├── sprl_model.py           # Spatial Role Labeling (trajector, landmark, indicator)
│   └── preposition_model.py    # Preposition sense disambiguation
│
├── network_translator/         # Linguistic-to-formal bridges
│   ├── gum_translator.py       # GUM ontology to FA-QSR constraints
│   └── sprl_to_network.py      # SpRL triples to constraint networks
│
├── feature_engineering/        # Feature extraction
│   └── spatial_features.py     # Affordances, context features
│
├── complexity_analysis/        # Computational complexity
│   └── complexity_analyzer.py  # Fragment classification, tractability
│
├── ontology/                   # OWL ontology files
│   ├── gum.owl                 # Generalized Upper Model
│   ├── gum-space.owl           # GUM spatial extension
│   └── faqsr_extension.owl     # FA-QSR ontology extension
│
├── data/                       # Sample data
│   └── sample_sentences.json   # Test sentences with annotations
│
├── results/                    # Evaluation and results
│   ├── empirical_analysis.py   # Empirical evaluation framework
│   ├── empirical_results.json  # Benchmark results
│   └── evaluation.py           # Evaluation utilities
│
├── tests/                      # Unit and integration tests
│   ├── test_rcc8.py            # RCC-8 tests
│   └── test_faqsr.py           # FA-QSR integration tests
│
└── latex_project/              # Academic paper
    ├── main.tex                # LaTeX source
    └── literature.bib          # Bibliography
```

## Module Documentation

### QSR Base (`qsr_base/`)

#### `rcc8.py` - Region Connection Calculus

Implements the RCC-8 spatial calculus with:
- **8 JEPD Relations**: DC, EC, PO, EQ, TPP, NTPP, TPPi, NTPPi
- **64-entry Composition Table**: Computes possible relations given intermediate constraints
- **H8 Tractable Fragment**: 148 relation sets solvable in polynomial time
- **Constraint Networks**: Graph-based representation of spatial configurations

```python
from qsr_base.rcc8 import RCC8Relation, compose, TractableFragments

# Compose relations: if A is inside B (NTPP), and B is inside C (NTPP)...
result = compose(RCC8Relation.NTPP, RCC8Relation.NTPP)
# result = {NTPP}  (A is inside C)

# Check if a relation set is tractable
is_tractable = TractableFragments.is_in_h8(relation_set)
```

#### `faqsr_calculus.py` - Functional Extension

Extends RCC-8 with functional primitives:
- **`FSUPPORT`**: Horizontal support (table-cup)
- **`FSUPPORT_VERTICAL`**: Vertical attachment (wall-picture)
- **`FSUPPORT_HANGING`**: Hanging support (hook-coat)
- **`FSUPPORT_ADHESION`**: Adhesive attachment (magnet-note)
- **`FCONTAIN`**: Full containment (box-object)
- **`FCONTAIN_PARTIAL`**: Partial containment (vase-flowers)
- **`FCONTAIN_PERMEABLE`**: Permeable containment (net-fish)

### NLP Models (`nlp_models/`)

#### `sprl_model.py` - Spatial Role Labeling

Extracts spatial triples from text:
- **Trajector**: The located entity ("the cup")
- **Landmark**: The reference entity ("the table")
- **Spatial Indicator**: The relational term ("on")

```python
from nlp_models.sprl_model import SpatialRoleLabeler

labeler = SpatialRoleLabeler()
triples = labeler.extract_triples(tokens)
# Returns: [SpatialTriple(trajector="cup", indicator="on", landmark="table")]
```

#### `preposition_model.py` - Sense Disambiguation

Disambiguates polysemous prepositions based on trajector-landmark context:

| Preposition | Sense | Example |
|-------------|-------|---------|
| on | support (horizontal) | cup on table |
| on | support (vertical) | picture on wall |
| on | support (hanging) | coat on hook |
| in | container | water in cup |
| in | region | cat in room |
| above | projective | lamp above desk |

```python
from nlp_models.preposition_model import PrepositionDisambiguator

disambiguator = PrepositionDisambiguator()
result = disambiguator.disambiguate(triple)
print(result.predicted_sense)  # PrepositionSense.ON_SUPPORT
print(result.is_functional)    # True
```

### Reasoning Engine (`reasoning_engine/`)

#### `path_consistency.py`

Implements constraint propagation:
- **Path Consistency Algorithm**: O(n³) iterations, refines constraints until fixed point
- **Consistency Checking**: Detects contradictory spatial configurations
- **Relation Inference**: Derives implied relations between entities

```python
from reasoning_engine.path_consistency import FAQSRReasoner

reasoner = FAQSRReasoner(max_iterations=10000)
result = reasoner.check_consistency(network)

if result.status == ConsistencyStatus.CONSISTENT:
    print("Spatial configuration is possible")
elif result.status == ConsistencyStatus.INCONSISTENT:
    print(f"Contradiction found: {result.conflict}")
```

#### `backtrack.py`

Backtracking search for NP-hard fragments:
- Variable and value ordering heuristics (MRV, LCV)
- Constraint propagation at each node
- Solution enumeration or satisfiability checking

### Complexity Analysis (`complexity_analysis/`)

Classifies constraint problems by computational complexity:

| Fragment Type | Tractable % | Avg. Time | Notes |
|---------------|-------------|-----------|-------|
| Pure Geometric | 100% | <1ms | H8 fragment, path consistency complete |
| Pure Functional | 8% | ~9ms | Specialized algorithms may help |
| Hybrid | 12% | ~9ms | Approaches NP-hardness |

```python
from complexity_analysis.complexity_analyzer import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()
result = analyzer.analyze(network)

print(f"Class: {result.complexity_class.value}")      # "polynomial" or "np-hard"
print(f"Tractable: {result.is_tractable}")            # True/False
print(f"Cognitive difficulty: {result.predicted_processing_difficulty}")  # 1-3
```

## Configuration

FA-QSR is configurable via the `FAQSRConfig` class:

```python
from faqsr import FAQSR, FAQSRConfig

config = FAQSRConfig(
    # NLP settings
    use_heuristic_parsing=True,
    use_semantic_features=True,

    # Reasoning settings
    use_path_consistency=True,
    max_reasoning_iterations=10000,
    enable_backtracking=False,  # Enable for NP-hard fragments

    # Complexity settings
    analyze_complexity=True,
    warn_on_np_hard=True,
)

faqsr = FAQSR(config)
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test module
python -m pytest tests/test_faqsr.py -v
```

## Running the Demo

```bash
python faqsr.py
```

This runs a demonstration showing:
1. Natural language processing of spatial text
2. Direct network construction
3. Processing pre-extracted triples
4. Consistency checking and inference
5. Complexity analysis

## Empirical Evaluation

Run the empirical analysis:

```bash
python results/empirical_analysis.py
```

This generates:
- `results/empirical_results.json`: Complete benchmark results
- LaTeX tables for academic publication
- Summary statistics

### Key Empirical Findings

- **Disambiguation Accuracy**: 86% overall (100% precision for functional interpretations)
- **Consistency Detection**: 100% accuracy
- **Cognitive Difficulty Correlation**: Functional prepositions (2.72/3) vs Geometric (1.68/3)

## Theoretical Background

FA-QSR draws on:

1. **Qualitative Spatial Reasoning** (Cohn & Renz, 2008): Region Connection Calculus and tractable fragments
2. **Linguistic Ontology** (Bateman et al., 2010): Generalized Upper Model spatial categories
3. **Functional Geometry** (Coventry et al., 2001, 2005): Interplay of geometric and functional factors
4. **Cognitive Evidence** (Landau, 2024): Dual nature of spatial terms (geometric vs. force-dynamic)

## Citation

If you use FA-QSR in your research, please cite:

```bibtex
@article{sparks2024faqsr,
  author    = {Sparks, Shane},
  title     = {Integrating Formal Ontology with Qualitative Spatial Reasoning},
  institution = {Indiana University Bloomington},
  year      = {2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- The RCC-8 composition table follows Randell, Cui, & Cohn (1992)
- The H8 tractable fragment analysis follows Renz & Nebel (1999)
- GUM ontology categories follow Bateman et al. (2010)
- Functional geometry insights from Coventry, Garrod, and colleagues
