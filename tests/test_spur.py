from typing import Any
from types import SimpleNamespace

import pandas as pd

import spur.core as pipeline
from spur import PipelineResult, spur


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
