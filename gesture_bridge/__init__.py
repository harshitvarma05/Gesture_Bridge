"""Core services for the Gesture-Bridge edge prototype."""

from .alerts import AlertManager
from .intelligence import ContextInterpreter, SentenceEngine
from .safety import SafetyAnalyzer

__all__ = ["AlertManager", "ContextInterpreter", "SentenceEngine", "SafetyAnalyzer"]
__version__ = "1.0.0-rc1"
