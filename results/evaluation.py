"""
Evaluation Framework for FA-QSR

This module provides comprehensive evaluation capabilities for the
FA-QSR system, including:

1. NLP Pipeline Evaluation
   - Spatial Role Labeling (SpRL) metrics
   - Preposition disambiguation accuracy

2. Reasoning Engine Evaluation
   - Consistency checking accuracy
   - Inference correctness
   - Performance benchmarking

3. Complexity Analysis Validation
   - Tractability prediction accuracy
   - Cognitive difficulty correlation

References:
- SpaceEval metrics (Pustejovsky et al., 2015)
- Standard IR metrics (precision, recall, F1)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsr_base.faqsr_calculus import FAQSRNetwork, FAQSRConstraint
from nlp_models.sprl_model import SpatialTriple, SpRLAnnotation
from reasoning_engine.path_consistency import ConsistencyResult, ConsistencyStatus


@dataclass
class SpRLMetrics:
    """Metrics for Spatial Role Labeling evaluation."""
    # Per-role metrics
    trajector_precision: float = 0.0
    trajector_recall: float = 0.0
    trajector_f1: float = 0.0

    landmark_precision: float = 0.0
    landmark_recall: float = 0.0
    landmark_f1: float = 0.0

    indicator_precision: float = 0.0
    indicator_recall: float = 0.0
    indicator_f1: float = 0.0

    # Overall metrics
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0

    # Triple-level metrics
    exact_match_accuracy: float = 0.0
    partial_match_accuracy: float = 0.0


@dataclass
class ReasoningMetrics:
    """Metrics for reasoning engine evaluation."""
    consistency_accuracy: float = 0.0
    inference_accuracy: float = 0.0
    avg_reasoning_time_ms: float = 0.0
    max_reasoning_time_ms: float = 0.0

    # Breakdown by complexity
    tractable_accuracy: float = 0.0
    intractable_accuracy: float = 0.0

    # Error analysis
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    sprl_metrics: Optional[SpRLMetrics] = None
    reasoning_metrics: Optional[ReasoningMetrics] = None

    # Metadata
    timestamp: str = ""
    dataset_name: str = ""
    num_examples: int = 0

    # Additional info
    notes: List[str] = field(default_factory=list)


class MetricsCollector:
    """
    Collects and aggregates metrics during evaluation.
    """

    def __init__(self):
        self._sprl_results: List[Dict] = []
        self._reasoning_results: List[Dict] = []
        self._timing_results: List[float] = []

    def record_sprl_result(self, predicted: List[SpRLAnnotation],
                          gold: List[SpRLAnnotation]) -> None:
        """Record a single SpRL prediction result."""
        result = {
            'predicted': len(predicted),
            'gold': len(gold),
            'matches': self._count_matches(predicted, gold)
        }
        self._sprl_results.append(result)

    def record_reasoning_result(self, predicted: bool, gold: bool,
                               time_ms: float) -> None:
        """Record a single reasoning result."""
        self._reasoning_results.append({
            'predicted': predicted,
            'gold': gold,
            'correct': predicted == gold,
            'time_ms': time_ms
        })
        self._timing_results.append(time_ms)

    def compute_sprl_metrics(self) -> SpRLMetrics:
        """Compute aggregated SpRL metrics."""
        if not self._sprl_results:
            return SpRLMetrics()

        total_predicted = sum(r['predicted'] for r in self._sprl_results)
        total_gold = sum(r['gold'] for r in self._sprl_results)
        total_matches = sum(r['matches'] for r in self._sprl_results)

        precision = total_matches / total_predicted if total_predicted > 0 else 0
        recall = total_matches / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return SpRLMetrics(
            overall_precision=precision,
            overall_recall=recall,
            overall_f1=f1
        )

    def compute_reasoning_metrics(self) -> ReasoningMetrics:
        """Compute aggregated reasoning metrics."""
        if not self._reasoning_results:
            return ReasoningMetrics()

        correct = sum(1 for r in self._reasoning_results if r['correct'])
        total = len(self._reasoning_results)

        fp = sum(1 for r in self._reasoning_results
                if r['predicted'] and not r['gold'])
        fn = sum(1 for r in self._reasoning_results
                if not r['predicted'] and r['gold'])

        avg_time = sum(self._timing_results) / len(self._timing_results) if self._timing_results else 0
        max_time = max(self._timing_results) if self._timing_results else 0

        return ReasoningMetrics(
            consistency_accuracy=correct / total if total > 0 else 0,
            avg_reasoning_time_ms=avg_time,
            max_reasoning_time_ms=max_time,
            false_positives=fp,
            false_negatives=fn
        )

    def _count_matches(self, predicted: List[SpRLAnnotation],
                      gold: List[SpRLAnnotation]) -> int:
        """Count matching annotations."""
        matches = 0
        for p in predicted:
            for g in gold:
                if self._annotations_match(p, g):
                    matches += 1
                    break
        return matches

    def _annotations_match(self, a: SpRLAnnotation, b: SpRLAnnotation) -> bool:
        """Check if two annotations match (with overlap tolerance)."""
        # Check trajector overlap
        traj_match = (a.trajector and b.trajector and
                     self._spans_overlap(a.trajector, b.trajector))

        # Check landmark overlap
        land_match = (a.landmark and b.landmark and
                     self._spans_overlap(a.landmark, b.landmark))

        # Check indicator overlap
        ind_match = (a.indicator and b.indicator and
                    self._spans_overlap(a.indicator, b.indicator))

        # All three must match for a complete match
        return traj_match and land_match and ind_match

    def _spans_overlap(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Check if two spans overlap."""
        return a[0] < b[1] and b[0] < a[1]

    def reset(self):
        """Reset collected results."""
        self._sprl_results = []
        self._reasoning_results = []
        self._timing_results = []


class Evaluator:
    """
    Main evaluator for FA-QSR system components.
    """

    def __init__(self):
        self.collector = MetricsCollector()

    def evaluate_sprl(self, predictions: List[List[SpRLAnnotation]],
                     gold_labels: List[List[SpRLAnnotation]]) -> SpRLMetrics:
        """
        Evaluate SpRL predictions against gold labels.

        Args:
            predictions: List of predicted annotations per example
            gold_labels: List of gold annotations per example

        Returns:
            SpRLMetrics with evaluation results
        """
        self.collector.reset()

        for pred, gold in zip(predictions, gold_labels):
            self.collector.record_sprl_result(pred, gold)

        return self.collector.compute_sprl_metrics()

    def evaluate_reasoning(self, networks: List[FAQSRNetwork],
                          gold_consistency: List[bool],
                          reasoner) -> ReasoningMetrics:
        """
        Evaluate reasoning engine predictions.

        Args:
            networks: List of FA-QSR networks to check
            gold_consistency: Gold standard consistency labels
            reasoner: FAQSRReasoner instance

        Returns:
            ReasoningMetrics with evaluation results
        """
        self.collector.reset()

        for network, gold in zip(networks, gold_consistency):
            start_time = time.time()
            result = reasoner.check_consistency(network)
            elapsed_ms = (time.time() - start_time) * 1000

            predicted = result.status == ConsistencyStatus.CONSISTENT
            self.collector.record_reasoning_result(predicted, gold, elapsed_ms)

        return self.collector.compute_reasoning_metrics()

    def full_evaluation(self,
                       sprl_predictions: Optional[List[List[SpRLAnnotation]]] = None,
                       sprl_gold: Optional[List[List[SpRLAnnotation]]] = None,
                       networks: Optional[List[FAQSRNetwork]] = None,
                       gold_consistency: Optional[List[bool]] = None,
                       reasoner=None,
                       dataset_name: str = "unknown") -> EvaluationResult:
        """
        Run complete evaluation on all components.

        Args:
            sprl_predictions: SpRL predictions (optional)
            sprl_gold: SpRL gold labels (optional)
            networks: Networks for reasoning eval (optional)
            gold_consistency: Gold consistency labels (optional)
            reasoner: Reasoner for reasoning eval (optional)
            dataset_name: Name of evaluation dataset

        Returns:
            Complete EvaluationResult
        """
        sprl_metrics = None
        reasoning_metrics = None
        num_examples = 0

        if sprl_predictions and sprl_gold:
            sprl_metrics = self.evaluate_sprl(sprl_predictions, sprl_gold)
            num_examples = len(sprl_predictions)

        if networks and gold_consistency and reasoner:
            reasoning_metrics = self.evaluate_reasoning(
                networks, gold_consistency, reasoner
            )
            num_examples = max(num_examples, len(networks))

        return EvaluationResult(
            sprl_metrics=sprl_metrics,
            reasoning_metrics=reasoning_metrics,
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name,
            num_examples=num_examples
        )


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    description: str = ""
    dataset_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """
    Runs and records FA-QSR experiments.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.experiments: List[Dict] = []

    def run_experiment(self, config: ExperimentConfig,
                      run_fn: Callable[[ExperimentConfig], EvaluationResult]) -> EvaluationResult:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration
            run_fn: Function that runs the experiment

        Returns:
            EvaluationResult from the experiment
        """
        print(f"Running experiment: {config.name}")
        start_time = time.time()

        result = run_fn(config)

        elapsed = time.time() - start_time

        # Record experiment
        experiment_record = {
            'name': config.name,
            'description': config.description,
            'parameters': config.parameters,
            'result': self._result_to_dict(result),
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }
        self.experiments.append(experiment_record)

        print(f"Experiment completed in {elapsed:.2f}s")
        return result

    def save_results(self, filename: str = "experiment_results.json"):
        """Save all experiment results to file."""
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.experiments, f, indent=2)

        print(f"Results saved to {filepath}")

    def _result_to_dict(self, result: EvaluationResult) -> Dict:
        """Convert EvaluationResult to dictionary."""
        d = {
            'timestamp': result.timestamp,
            'dataset_name': result.dataset_name,
            'num_examples': result.num_examples,
            'notes': result.notes
        }

        if result.sprl_metrics:
            d['sprl_metrics'] = {
                'overall_precision': result.sprl_metrics.overall_precision,
                'overall_recall': result.sprl_metrics.overall_recall,
                'overall_f1': result.sprl_metrics.overall_f1,
            }

        if result.reasoning_metrics:
            d['reasoning_metrics'] = {
                'consistency_accuracy': result.reasoning_metrics.consistency_accuracy,
                'avg_reasoning_time_ms': result.reasoning_metrics.avg_reasoning_time_ms,
                'false_positives': result.reasoning_metrics.false_positives,
                'false_negatives': result.reasoning_metrics.false_negatives,
            }

        return d

    def generate_report(self) -> str:
        """Generate a summary report of all experiments."""
        lines = ["FA-QSR Experiment Report", "=" * 50, ""]

        for exp in self.experiments:
            lines.append(f"Experiment: {exp['name']}")
            lines.append(f"  Description: {exp.get('description', 'N/A')}")
            lines.append(f"  Time: {exp['elapsed_seconds']:.2f}s")

            if 'sprl_metrics' in exp['result']:
                m = exp['result']['sprl_metrics']
                lines.append(f"  SpRL F1: {m['overall_f1']:.3f}")

            if 'reasoning_metrics' in exp['result']:
                m = exp['result']['reasoning_metrics']
                lines.append(f"  Reasoning Accuracy: {m['consistency_accuracy']:.3f}")
                lines.append(f"  Avg Time: {m['avg_reasoning_time_ms']:.2f}ms")

            lines.append("")

        return "\n".join(lines)


def demonstrate_evaluation():
    """Demonstrate evaluation framework."""
    print("FA-QSR Evaluation Framework Demonstration")
    print("=" * 50)

    # Create sample data
    from feature_engineering.spatial_features import Token
    from nlp_models.sprl_model import SpatialRoleLabeler

    # Sample annotations (simulated)
    gold_annotations = [
        [SpRLAnnotation(
            trajector=(1, 2),
            landmark=(4, 5),
            indicator=(2, 3),
            trajector_text="book",
            landmark_text="table",
            indicator_text="on"
        )]
    ]

    pred_annotations = [
        [SpRLAnnotation(
            trajector=(1, 2),
            landmark=(4, 5),
            indicator=(2, 3),
            trajector_text="book",
            landmark_text="table",
            indicator_text="on"
        )]
    ]

    # Evaluate SpRL
    evaluator = Evaluator()
    sprl_metrics = evaluator.evaluate_sprl(pred_annotations, gold_annotations)

    print("\nSpRL Evaluation Results:")
    print(f"  Precision: {sprl_metrics.overall_precision:.3f}")
    print(f"  Recall: {sprl_metrics.overall_recall:.3f}")
    print(f"  F1: {sprl_metrics.overall_f1:.3f}")

    # Run experiment
    print("\nExperiment Runner Demo:")
    runner = ExperimentRunner()

    def dummy_experiment(config):
        return EvaluationResult(
            sprl_metrics=sprl_metrics,
            timestamp=datetime.now().isoformat(),
            dataset_name="demo",
            num_examples=1
        )

    config = ExperimentConfig(
        name="demo_experiment",
        description="Demonstration experiment"
    )

    result = runner.run_experiment(config, dummy_experiment)

    print("\nExperiment Report:")
    print(runner.generate_report())


if __name__ == "__main__":
    demonstrate_evaluation()
