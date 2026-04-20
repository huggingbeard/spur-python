from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SpurTestResult:
    """Container for spurtest results."""

    test_type: str
    LR: float
    pvalue: float
    cv: np.ndarray
    ha_param: float

    def summary(self) -> str:
        """Format test results for display."""
        stat_name = "LFUR" if self.test_type.startswith("i1") else "LFST"
        lines = [
            f"Spatial {self.test_type.upper()} Test Results",
            "-" * 45,
            f"Test Statistic ({stat_name}):  {self.LR:9.4f}",
            f"P-value:                {self.pvalue:9.4f}",
            f"CV 1%:                  {self.cv[0]:9.4f}",
            f"CV 5%:                  {self.cv[1]:9.4f}",
            f"CV 10%:                 {self.cv[2]:9.4f}",
            "-" * 45,
        ]
        return "\n".join(lines)


@dataclass
class HalfLifeResult:
    """Container for spurhalflife results."""

    ci_lower: float
    ci_upper: float
    max_dist: float
    level: float
    normdist: bool

    def summary(self) -> str:
        """Format results for display."""
        units = "fractions of max distance" if self.normdist else "meters"
        upper_str = "inf" if np.isinf(self.ci_upper) else f"{self.ci_upper:.4f}"
        lines = [
            f"Spatial half-life {self.level:g}% confidence interval ({units})",
            "-" * 45,
            f"Lower bound: {self.ci_lower:.4f}",
            f"Upper bound: {upper_str}",
            f"Max distance in sample: {self.max_dist:.4f}",
            "-" * 45,
        ]
        return "\n".join(lines)


@dataclass
class SpurResult:
    """Container for the high-level SPUR pipeline output."""

    branch: str
    test_i0: SpurTestResult
    test_i1: SpurTestResult
    model: object
    scpc: object
    data_used: pd.DataFrame
    formula_used: str
