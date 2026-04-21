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


def format_comparison_row(
    label: str,
    left: str,
    right: str,
    *,
    label_width: int,
    value_width: int,
    gap: int,
) -> str:
    """Format one row of the two-column regression table."""
    pad = " " * gap
    return f"{label:<{label_width}}{pad}{left:>{value_width}}{pad}{right:>{value_width}}"


def format_comparison_header_row(
    label: str,
    left: str,
    right: str,
    *,
    label_width: int,
    value_width: int,
    gap: int,
) -> str:
    """Format one header row for the two-column table layout."""
    pad = " " * gap
    return f"{label:<{label_width}}{pad}{left:<{value_width}}{pad}{right:<{value_width}}"


def collect_coefficient_rows(result: PipelineResult) -> list[tuple[str, str, str]]:
    """Collect formatted coefficient and standard-error rows."""
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

    coefficient_rows: list[tuple[str, str, str]] = []
    for name in row_order:
        levels_pair = levels_rows.get(name)
        transformed_pair = transformed_rows.get(name)

        coefficient_rows.append(
            (
                name,
                format_decimal(levels_pair[0]) if levels_pair is not None else "",
                format_decimal(transformed_pair[0])
                if transformed_pair is not None
                else "",
            )
        )
        coefficient_rows.append(
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

    return coefficient_rows


def collect_model_stat_rows(result: PipelineResult) -> list[tuple[str, str, str]]:
    """Collect formatted model-statistic rows."""
    return [
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


def collect_diagnostic_rows(
    result: PipelineResult,
) -> list[tuple[str, str, str]]:
    """Collect formatted SPUR diagnostic rows."""
    return [
        (
            "i0",
            format_decimal(result.tests.i0.LR),
            format_decimal(result.tests.i0.pvalue),
        ),
        (
            "i1",
            format_decimal(result.tests.i1.LR),
            format_decimal(result.tests.i1.pvalue),
        ),
        (
            "i0resid",
            format_decimal(result.tests.i0resid.LR),
            format_decimal(result.tests.i0resid.pvalue),
        ),
        (
            "i1resid",
            format_decimal(result.tests.i1resid.LR),
            format_decimal(result.tests.i1resid.pvalue),
        ),
    ]


def render_diagnostics_section(
    diagnostic_rows: list[tuple[str, str, str]],
    *,
    table_width: int,
    label_width: int,
    value_width: int,
    gap: int,
    rule: str,
) -> list[str]:
    """Render the standalone diagnostics block."""
    lines: list[str] = ["SPUR Diagnostics".center(table_width), rule]
    lines.append(
        format_comparison_header_row(
            "Test",
            "LR",
            "p-value",
            label_width=label_width,
            value_width=value_width,
            gap=gap,
        ).ljust(table_width)
    )
    for name, lr, pvalue in diagnostic_rows:
        lines.append(
            format_comparison_row(
                name,
                lr,
                pvalue,
                label_width=label_width,
                value_width=value_width,
                gap=gap,
            ).ljust(table_width)
        )
    return lines


def render_regression_section(
    dependent_variable: str,
    coefficient_rows: list[tuple[str, str, str]],
    model_stat_rows: list[tuple[str, str, str]],
    *,
    table_width: int,
    label_width: int,
    value_width: int,
    gap: int,
    rule: str,
) -> list[str]:
    """Render the regression comparison table and model statistics."""
    span_width = value_width + gap + value_width
    span_indent = label_width + gap
    lines: list[str] = [
        (" " * span_indent + dependent_variable.center(span_width)).ljust(table_width),
        (" " * span_indent + "-" * span_width).ljust(table_width),
        format_comparison_header_row(
            "Coefficient",
            "Levels",
            "Transformed",
            label_width=label_width,
            value_width=value_width,
            gap=gap,
        ).ljust(table_width),
        rule,
    ]
    for name, left, right in coefficient_rows:
        lines.append(
            format_comparison_row(
                name,
                left,
                right,
                label_width=label_width,
                value_width=value_width,
                gap=gap,
            ).ljust(table_width)
        )
    lines.append(rule)
    for name, left, right in model_stat_rows:
        lines.append(
            format_comparison_row(
                name,
                left,
                right,
                label_width=label_width,
                value_width=value_width,
                gap=gap,
            ).ljust(table_width)
        )
    return lines


def render_pipeline_summary(result: PipelineResult) -> str:
    """Render a centered summary table for `PipelineResult`."""
    dependent_variable = str(result.fits.levels.model.model.endog_names)
    coefficient_rows = collect_coefficient_rows(result)
    model_stat_rows = collect_model_stat_rows(result)
    diagnostic_rows = collect_diagnostic_rows(result)

    comparison_gap = 4
    comparison_label_width = max(
        len("Coefficient"),
        len("Adj. R-squared"),
        len("SCPC avc"),
        *(len(name) for name, _, _ in coefficient_rows),
    )
    diagnostics_label_width = max(
        len("Test"),
        *(len(name) for name, _, _ in diagnostic_rows),
    )
    label_width = max(comparison_label_width, diagnostics_label_width)
    comparison_value_width = max(
        len("Levels"),
        len("Transformed"),
        *[
            len(value)
            for _, left, right in [*coefficient_rows, *model_stat_rows]
            for value in (left, right)
        ],
    )
    dependent_width = comparison_value_width + comparison_gap + comparison_value_width
    if len(dependent_variable) > dependent_width:
        comparison_value_width = max(
            comparison_value_width,
            (len(dependent_variable) - comparison_gap + 1) // 2,
        )

    comparison_span_width = comparison_value_width + comparison_gap + comparison_value_width

    table_width = label_width + comparison_gap + comparison_span_width
    rule = "-" * table_width

    lines: list[str] = [rule, rule]
    lines.extend(
        render_diagnostics_section(
            diagnostic_rows,
            table_width=table_width,
            label_width=label_width,
            value_width=comparison_value_width,
            gap=comparison_gap,
            rule=rule,
        )
    )
    lines.extend(
        [rule, rule, "", "Regression results".center(table_width), rule, rule]
    )
    lines.extend(
        render_regression_section(
            dependent_variable,
            coefficient_rows,
            model_stat_rows,
            table_width=table_width,
            label_width=label_width,
            value_width=comparison_value_width,
            gap=comparison_gap,
            rule=rule,
        )
    )
    lines.extend([rule, rule])

    return "\n".join(lines)
