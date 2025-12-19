"""
FA-QSR: Functionally-Augmented Qualitative Spatial Reasoning

A framework for bridging linguistically motivated spatial categories
from the Generalized Upper Model (GUM) with algebraic constraint-based
reasoning from Qualitative Spatial Reasoning (QSR).

Key Features:
- Functional spatial primitives (fsupport, fcontainment)
- RCC-8 based geometric reasoning
- Hybrid geometric-functional constraint networks
- Natural language to constraint translation
- Complexity-aware reasoning

Example:
    from faqsr import FAQSR

    faqsr = FAQSR()
    result = faqsr.process("The book is on the table")
    print(result.triples)
"""

__version__ = "1.0.0"
__author__ = "FA-QSR Project"

from .faqsr import (
    FAQSR,
    FAQSRConfig,
    ProcessingResult,
    InferenceResult,
    create_example_network,
)

__all__ = [
    'FAQSR',
    'FAQSRConfig',
    'ProcessingResult',
    'InferenceResult',
    'create_example_network',
]
