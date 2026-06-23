"""Core services for the Gesture-Bridge edge prototype."""

from .alerts import AlertManager
from .intelligence import ContextInterpreter, SentenceEngine
from .safety import SafetyAnalyzer
from .recognition import GestureDebouncer

__all__ = ["AlertManager", "ContextInterpreter", "SentenceEngine", "SafetyAnalyzer", "GestureDebouncer"]
__version__ = "1.0.0-rc1"
