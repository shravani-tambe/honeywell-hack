"""Rationale Text for setpoint suggestions

Plain-language sentences built from the sensitivity-analysis and
correlation-discovery outputs, so the dashboard never needs a hand-written
explanation per suggestion.
"""

TREND_TEMPLATE = (
    "Basis weight is running {trend} setpoint ({bw_error_pct:+.2f}% error). "
    "{action} {variable} by {delta_pct:.1f}% (from {current:.1f} to {recommended:.1f}) "
    "is predicted to cut off-spec risk by {reduction_pp:.1f} points."
)

CORRELATION_SUFFIX = (
    " Backed by a discovered correlation: {variable} {lag}s earlier tracks "
    "basis_weight now (r={correlation:.2f})."
)

CLIP_SUFFIX = " Capped at the recipe operating limit for {variable}."


def build_rationale(variable, current, recommended, reduction, bw_error_pct, correlation=None, clipped=False):
    trend = "above" if bw_error_pct >= 0 else "below"
    action = "Increasing" if recommended > current else "Decreasing"
    delta_pct = abs(recommended - current) / current * 100 if current else 0.0

    text = TREND_TEMPLATE.format(
        trend=trend,
        bw_error_pct=bw_error_pct,
        action=action,
        variable=variable,
        delta_pct=delta_pct,
        current=current,
        recommended=recommended,
        reduction_pp=reduction * 100,
    )
    if correlation:
        text += CORRELATION_SUFFIX.format(
            variable=variable, lag=correlation["lag"], correlation=correlation["correlation"]
        )
    if clipped:
        text += CLIP_SUFFIX.format(variable=variable)
    return text
