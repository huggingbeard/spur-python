from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(slots=True)
class TestResult:
    """One SPUR test result."""

    __test__ = False

    test_type: str
    """Name of the test that was run."""
    LR: float
    """Observed value of the test statistic."""
    pvalue: float
    """P-value associated with the test statistic."""
    cv: np.ndarray
    """Reference critical values for the test."""
    ha_param: float
    """Estimated local-alternative tuning parameter."""

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
    """Spatial half-life interval result."""

    ci_lower: float
    """Lower end of the reported interval."""
    ci_upper: float
    """Upper end of the reported interval."""
    max_dist: float
    """Largest observed distance in the sample."""
    level: float
    """Confidence level used for the interval."""
    normdist: bool
    """Whether distances were normalized before reporting."""

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


@dataclass(slots=True)
class Tests:
    """The four SPUR diagnostics returned by `spur()`."""

    __test__ = False

    i0: TestResult
    """I(0) test run on the dependent variable in levels."""
    i1: TestResult
    """I(1) test run on the dependent variable in levels."""
    i0resid: TestResult
    """I(0) test run on the regression residuals."""
    i1resid: TestResult
    """I(1) test run on the regression residuals."""


@dataclass(slots=True)
class RegressionResult:
    model: Any
    """Fitted regression model for this branch."""
    scpc: Any
    """SCPC output computed from that fitted model."""


@dataclass(slots=True)
class Fits:
    """The two fitted regression branches returned by `spur()`."""

    levels: RegressionResult
    """Fit based on the original variables."""
    transformed: RegressionResult
    """Fit based on the transformed variables."""


@dataclass(slots=True)
class PipelineResult:
    """Full output of `spur()`."""

    tests: Tests
    """All diagnostic test results."""
    fits: Fits
    """Both fitted regression branches."""

    def summary(self) -> str:
        """Format the full SPUR pipeline result for display."""
        from .utils.summary import render_pipeline_summary

        return render_pipeline_summary(self)
