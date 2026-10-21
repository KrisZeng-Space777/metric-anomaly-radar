"""metric-anomaly-radar: anomaly detection + dimension attribution for ad metrics."""

from __future__ import annotations

from . import anomalies, detectors, synth

__version__ = "0.1.0"

__all__ = ["anomalies", "detectors", "synth", "__version__"]
