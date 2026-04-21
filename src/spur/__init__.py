from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from .core import (
    spur as spur_fn,
    spurhalflife,
    spurtest,
    spurtest_i0,
    spurtest_i0resid,
    spurtest_i1,
    spurtest_i1resid,
    spurtransform,
)
from .types import (
    Fits,
    HalfLifeResult,
    PipelineResult,
    RegressionResult,
    TestResult,
    Tests,
)
from .utils.data import load_chetty_data, standardize

spur = spur_fn


class SPUR(ModuleType):
    """Module type forwarding calls to `spur()`."""

    def __call__(self, *args: Any, **kwargs: Any) -> PipelineResult:
        return self.spur(*args, **kwargs)


module = sys.modules[__name__]
module.__class__ = SPUR

__all__ = [
    "load_chetty_data",
    "standardize",
    "HalfLifeResult",
    "TestResult",
    "RegressionResult",
    "Tests",
    "Fits",
    "PipelineResult",
    "spurtest_i1",
    "spurtest_i0",
    "spurtest_i1resid",
    "spurtest_i0resid",
    "spurtest",
    "spurtransform",
    "spurhalflife",
    "spur",
]
