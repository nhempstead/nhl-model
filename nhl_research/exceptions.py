from __future__ import annotations


class NHLResearchError(Exception):
    """Base error for the research platform."""


class LeakageError(NHLResearchError):
    """Raised when a point-in-time constraint would be violated."""


class DataUnavailableError(NHLResearchError):
    """Raised when a required dataset or field is not available."""


class InvalidOddsError(NHLResearchError):
    """Raised for American or decimal odds that cannot be converted."""


class PromotionRejected(NHLResearchError):
    """Raised when a challenger fails predefined promotion tests."""
