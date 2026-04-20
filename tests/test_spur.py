from types import SimpleNamespace

import pandas as pd

import spur.pipeline as pipeline
from spur import SpurResult, spur


def test_spur_uses_levels_branch_at_10_percent(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "lon": [10.0, 11.0, 12.0, 13.0],
            "lat": [50.0, 51.0, 52.0, 53.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [2.0, 1.0, 0.0, -1.0],
        }
    )

    monkeypatch.setattr(
        pipeline,
        "spurtest_i0",
        lambda *args, **kwargs: SimpleNamespace(pvalue=0.10),
    )
    monkeypatch.setattr(
        pipeline,
        "spurtest_i1",
        lambda *args, **kwargs: SimpleNamespace(pvalue=0.09),
    )

    calls: dict[str, object] = {}

    def fake_scpc(model, data, **kwargs):
        calls["formula"] = model.model.formula
        calls["rows"] = len(data)
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(pipeline, "scpc", fake_scpc)

    result = spur("y ~ x", df, lon="lon", lat="lat", q=10, nrep=200, seed=42)

    assert isinstance(result, SpurResult)
    assert result.branch == "levels"
    assert result.formula_used == "y ~ x"
    assert calls["formula"] == "y ~ x"
    assert calls["rows"] == len(df)


def test_spur_uses_transformed_branch_otherwise(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "lon": [10.0, 11.0, 12.0, 13.0],
            "lat": [50.0, 51.0, 52.0, 53.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [2.0, 1.0, 0.0, -1.0],
        }
    )

    monkeypatch.setattr(
        pipeline,
        "spurtest_i0",
        lambda *args, **kwargs: SimpleNamespace(pvalue=0.05),
    )
    monkeypatch.setattr(
        pipeline,
        "spurtest_i1",
        lambda *args, **kwargs: SimpleNamespace(pvalue=0.20),
    )
    monkeypatch.setattr(
        pipeline,
        "spurtransform",
        lambda *args, **kwargs: df.assign(h_y=df["y"], h_x=df["x"]),
    )

    calls: dict[str, object] = {}

    def fake_scpc(model, data, **kwargs):
        calls["formula"] = model.model.formula
        calls["cols"] = list(data.columns)
        return {"ok": True}

    monkeypatch.setattr(pipeline, "scpc", fake_scpc)

    result = spur("y ~ x", df, lon="lon", lat="lat", q=10, nrep=200, seed=42)

    assert isinstance(result, SpurResult)
    assert result.branch == "transformed"
    assert result.formula_used == "h_y ~ h_x"
    assert calls["formula"] == "h_y ~ h_x"
    assert "h_y" in calls["cols"]
    assert "h_x" in calls["cols"]
