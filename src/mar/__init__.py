"""metric-anomaly-radar: anomaly detection + dimension attribution for ad metrics."""

from __future__ import annotations

from . import anomalies, benchmark, detectors, evaluation, synth

__version__ = "0.1.0"

__all__ = ["anomalies", "benchmark", "detectors", "evaluation", "synth", "__version__"]
