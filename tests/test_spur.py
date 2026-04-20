from typing import Any
from types import SimpleNamespace

import numpy as np
import pandas as pd

import spur.core as pipeline
from spur import (
    Fits,
    PipelineResult,
    RegressionResult,
    TestResult,
    Tests,
    spur,
)


def test_spur_returns_full_pipeline_result(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "lon": [10.0, 11.0, 12.0, 13.0],
            "lat": [50.0, 51.0, 52.0, 53.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [2.0, 1.0, 0.0, -1.0],
        }
    )

    i0 = SimpleNamespace(name="i0")
    i1 = SimpleNamespace(name="i1")
    i0resid = SimpleNamespace(name="i0resid")
    i1resid = SimpleNamespace(name="i1resid")

    monkeypatch.setattr(pipeline, "spurtest_i0", lambda *args, **kwargs: i0)
    monkeypatch.setattr(pipeline, "spurtest_i1", lambda *args, **kwargs: i1)
    monkeypatch.setattr(pipeline, "spurtest_i0resid", lambda *args, **kwargs: i0resid)
    monkeypatch.setattr(pipeline, "spurtest_i1resid", lambda *args, **kwargs: i1resid)
    monkeypatch.setattr(
        pipeline,
        "spurtransform",
        lambda *args, **kwargs: df.assign(h_y=df["y"], h_x=df["x"]),
    )

    scpc_calls: list[dict[str, Any]] = []

    def fake_scpc(model, data, **kwargs):
        call = {
            "formula": model.model.formula,
            "rows": len(data),
            "cols": list(data.columns),
            "kwargs": kwargs,
        }
        scpc_calls.append(call)
        return call

    monkeypatch.setattr(pipeline, "scpc", fake_scpc)

    result = spur("y ~ x", df, lon="lon", lat="lat", q=10, nrep=200, seed=42)

    assert isinstance(result, PipelineResult)
    assert result.tests.i0 is i0
    assert result.tests.i1 is i1
    assert result.tests.i0resid is i0resid
    assert result.tests.i1resid is i1resid
    assert result.fits.levels.model.model.formula == "y ~ x"
    assert result.fits.transformed.model.model.formula == "h_y ~ h_x"
    assert result.fits.levels.scpc["formula"] == "y ~ x"
    assert result.fits.transformed.scpc["formula"] == "h_y ~ h_x"
    assert len(scpc_calls) == 2


def test_spur_passes_coordinate_kwargs_to_both_scpc_calls(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "lon": [10.0, 11.0, 12.0, 13.0],
            "lat": [50.0, 51.0, 52.0, 53.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [2.0, 1.0, 0.0, -1.0],
        }
    )

    monkeypatch.setattr(
        pipeline, "spurtest_i0", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        pipeline, "spurtest_i1", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        pipeline, "spurtest_i0resid", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        pipeline, "spurtest_i1resid", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        pipeline,
        "spurtransform",
        lambda *args, **kwargs: df.assign(h_y=df["y"], h_x=df["x"]),
    )

    scpc_calls: list[dict[str, Any]] = []

    def fake_scpc(model, data, **kwargs):
        scpc_calls.append(
            {
                "formula": model.model.formula,
                "cols": list(data.columns),
                "kwargs": kwargs,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(pipeline, "scpc", fake_scpc)

    spur("y ~ x", df, lon="lon", lat="lat", q=10, nrep=200, seed=42)

    assert len(scpc_calls) == 2
    assert scpc_calls[0]["kwargs"]["lon"] == "lon"
    assert scpc_calls[0]["kwargs"]["lat"] == "lat"
    assert scpc_calls[0]["kwargs"]["coords_euclidean"] is None
    assert scpc_calls[1]["kwargs"]["lon"] == "lon"
    assert scpc_calls[1]["kwargs"]["lat"] == "lat"
    assert scpc_calls[1]["kwargs"]["coords_euclidean"] is None
    assert scpc_calls[0]["formula"] == "y ~ x"
    assert scpc_calls[1]["formula"] == "h_y ~ h_x"
    assert "h_y" in scpc_calls[1]["cols"]
    assert "h_x" in scpc_calls[1]["cols"]


def test_pipeline_result_summary_formats_comparison_table() -> None:
    levels_model = SimpleNamespace(
        params=pd.Series([1.0, 0.5], index=["Intercept", "x"]),
        nobs=100.0,
        rsquared=0.2500,
        rsquared_adj=0.2300,
    )
    transformed_model = SimpleNamespace(
        params=pd.Series([0.9, 0.4, 0.2], index=["Intercept", "h_x", "h_z"]),
        nobs=100.0,
        rsquared=0.2100,
        rsquared_adj=0.1900,
    )

    result = PipelineResult(
        tests=Tests(
            i0=TestResult("i0", 1.1000, 0.2100, np.array([1.0, 2.0, 3.0]), 0.0100),
            i1=TestResult("i1", 1.2000, 0.1100, np.array([1.0, 2.0, 3.0]), 0.0200),
            i0resid=TestResult(
                "i0resid", 1.3000, 0.3100, np.array([1.0, 2.0, 3.0]), 0.0300
            ),
            i1resid=TestResult(
                "i1resid", 1.4000, 0.4100, np.array([1.0, 2.0, 3.0]), 0.0400
            ),
        ),
        fits=Fits(
            levels=RegressionResult(
                model=levels_model,
                scpc=SimpleNamespace(
                    scpcstats=np.array(
                        [
                            [1.0000, 0.2000, 5.0000, 0.0100, 0.6000, 1.4000],
                            [0.5000, 0.1000, 5.0000, 0.0200, 0.3000, 0.7000],
                        ]
                    ),
                    q=15,
                    cv=2.1034,
                    avc=0.0300,
                ),
            ),
            transformed=RegressionResult(
                model=transformed_model,
                scpc=SimpleNamespace(
                    scpcstats=np.array(
                        [
                            [0.9000, 0.1500, 6.0000, 0.0100, 0.6000, 1.2000],
                            [0.4000, 0.0500, 8.0000, 0.0200, 0.3000, 0.5000],
                            [0.2000, 0.0300, 6.7000, 0.0300, 0.1400, 0.2600],
                        ]
                    ),
                    q=12,
                    cv=2.0010,
                    avc=0.0300,
                ),
            ),
        ),
    )

    text = result.summary()

    assert "SPUR Pipeline Results" in text
    assert "Levels" in text
    assert "Transformed" in text
    assert "Intercept" in text
    assert "x" in text
    assert "z" in text
    assert "h_x" not in text
    assert "h_z" not in text
    assert "(0.2000)" in text
    assert "(0.0500)" in text
    assert "Model statistics" in text
    assert "R-squared" in text
    assert "Adj. R-squared" in text
    assert "SCPC q" in text
    assert "SCPC cv" in text
    assert "SCPC avc" in text
    assert "SPUR diagnostics" in text
    assert "i0" in text
    assert "i1" in text
    assert "i0resid" in text
    assert "i1resid" in text
