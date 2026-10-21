"""Detector registry."""

from __future__ import annotations

from .base import Detector, as_flags
from .cusum import CUSUMDetector
from .stl import STLDetector
from .zscore import RobustZDetector

ALL_DETECTORS = (RobustZDetector, STLDetector, CUSUMDetector)

__all__ = [
    "ALL_DETECTORS",
    "CUSUMDetector",
    "Detector",
    "RobustZDetector",
    "STLDetector",
    "as_flags",
]
