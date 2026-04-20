from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import PipelineResult


def display_term_name(name: str) -> str:
    """Return the coefficient label shown in the summary table."""
    return name[2:] if name.startswith("h_") else name


def format_decimal(value: float) -> str:
    """Format a decimal statistic for display."""
    return f"{float(value):.4f}"


def format_count(value: float) -> str:
    """Format a count statistic for display."""
    return str(int(round(float(value))))


def render_pipeline_summary(result: PipelineResult) -> str:
    """Render a statsmodels-like summary for `PipelineResult`."""
    levels_names = [str(name) for name in result.fits.levels.model.params.index]
    transformed_names = [
        display_term_name(str(name))
        for name in result.fits.transformed.model.params.index
    ]

    levels_rows = {
        name: (float(row[0]), float(row[1]))
        for name, row in zip(
            levels_names,
            result.fits.levels.scpc.scpcstats,
            strict=True,
        )
    }
    transformed_rows = {
        name: (float(row[0]), float(row[1]))
        for name, row in zip(
            transformed_names,
            result.fits.transformed.scpc.scpcstats,
            strict=True,
        )
    }

    row_order = list(levels_rows)
    for name in transformed_rows:
        if name not in levels_rows:
            row_order.append(name)

    coefficient_lines: list[tuple[str, str, str]] = []
    for name in row_order:
        levels_pair = levels_rows.get(name)
        transformed_pair = transformed_rows.get(name)

        coefficient_lines.append(
            (
                name,
                format_decimal(levels_pair[0]) if levels_pair is not None else "",
                format_decimal(transformed_pair[0])
                if transformed_pair is not None
                else "",
            )
        )
        coefficient_lines.append(
            (
                "",
                f"({format_decimal(levels_pair[1])})"
                if levels_pair is not None
                else "",
                f"({format_decimal(transformed_pair[1])})"
                if transformed_pair is not None
                else "",
            )
        )

    model_stats = [
        (
            "N",
            format_count(result.fits.levels.model.nobs),
            format_count(result.fits.transformed.model.nobs),
        ),
        (
            "R-squared",
            format_decimal(result.fits.levels.model.rsquared),
            format_decimal(result.fits.transformed.model.rsquared),
        ),
        (
            "Adj. R-squared",
            format_decimal(result.fits.levels.model.rsquared_adj),
            format_decimal(result.fits.transformed.model.rsquared_adj),
        ),
        (
            "SCPC q",
            format_count(result.fits.levels.scpc.q),
            format_count(result.fits.transformed.scpc.q),
        ),
        (
            "SCPC cv",
            format_decimal(result.fits.levels.scpc.cv),
            format_decimal(result.fits.transformed.scpc.cv),
        ),
        (
            "SCPC avc",
            format_decimal(result.fits.levels.scpc.avc),
            format_decimal(result.fits.transformed.scpc.avc),
        ),
    ]

    diagnostics = [
        (
            "i0",
            format_decimal(result.tests.i0.LR),
            format_decimal(result.tests.i0.pvalue),
            format_decimal(result.tests.i0.ha_param),
        ),
        (
            "i1",
            format_decimal(result.tests.i1.LR),
            format_decimal(result.tests.i1.pvalue),
            format_decimal(result.tests.i1.ha_param),
        ),
        (
            "i0resid",
            format_decimal(result.tests.i0resid.LR),
            format_decimal(result.tests.i0resid.pvalue),
            format_decimal(result.tests.i0resid.ha_param),
        ),
        (
            "i1resid",
            format_decimal(result.tests.i1resid.LR),
            format_decimal(result.tests.i1resid.pvalue),
            format_decimal(result.tests.i1resid.ha_param),
        ),
    ]

    label_width = max(
        len("Coefficient"),
        len("Adj. R-squared"),
        len("SCPC avc"),
        *(len(name) for name in row_order),
    )
    value_width = max(
        len("Levels"),
        len("Transformed"),
        *[
            len(value)
            for _, left, right in [*coefficient_lines, *model_stats]
            for value in (left, right)
        ],
    )

    diag_label_width = max(len("test"), *(len(name) for name, _, _, _ in diagnostics))
    diag_value_width = max(
        len("p-value"),
        len("ha_param"),
        *[
            len(value)
            for _, lr, pvalue, ha_param in diagnostics
            for value in (lr, pvalue, ha_param)
        ],
    )

    lines = [
        "SPUR Pipeline Results",
        "=====================",
        "",
        f"{'Coefficient':<{label_width}}  {'Levels':>{value_width}}  {'Transformed':>{value_width}}",
    ]
    for name, left, right in coefficient_lines:
        lines.append(
            f"{name:<{label_width}}  {left:>{value_width}}  {right:>{value_width}}"
        )

    lines.extend(["", "Model statistics", "----------------"])
    for name, left, right in model_stats:
        lines.append(
            f"{name:<{label_width}}  {left:>{value_width}}  {right:>{value_width}}"
        )

    lines.extend(["", "SPUR diagnostics", "----------------"])
    lines.append(
        f"{'test':<{diag_label_width}}  {'LR':>{diag_value_width}}  {'p-value':>{diag_value_width}}  {'ha_param':>{diag_value_width}}"
    )
    for name, lr, pvalue, ha_param in diagnostics:
        lines.append(
            f"{name:<{diag_label_width}}  {lr:>{diag_value_width}}  {pvalue:>{diag_value_width}}  {ha_param:>{diag_value_width}}"
        )

    return "\n".join(lines)
