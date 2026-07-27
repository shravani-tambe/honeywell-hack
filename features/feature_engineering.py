"""Feature engineering for Grade Change Intelligence.

Reads the raw simulator output and writes data/processed/features.csv with:
  - bw_error_pct           signed basis-weight deviation (direction for the recommender)
  - off_spec_future        forward-looking off-spec label (predict-before-it-happens target)
  - rolling mean/std       despiked, 30s/60s windows, for basis_weight/moisture/stock_flow
  - *_roc                  rate of change for the manipulated variables
  - *_lag{10,30,60}        transport-delay lags for stock_flow/machine_speed
  - transition_id          groups rows by simulator-generated transition (0-49) for
                           the group-aware train/test split

transition_scenario passed through unchanged and not as a model
feature (simulation ground truth); for stratified evaluation
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/paper_making_dataset_enhanced.csv"
OUTPUT_PATH = "data/processed/features.csv"

ROLLING_VARS = ["basis_weight", "moisture", "stock_flow"]
ROLLING_WINDOWS = [30, 60]
DESPIKE_VARS = ["basis_weight", "moisture"]  # only vars with injected sensor spikes/freezes
MANIPULATED_VARS = ["stock_flow", "filler_flow", "machine_speed", "steam_pressure"]
LAG_VARS = ["stock_flow", "machine_speed"]
LAG_OFFSETS = [10, 30, 60]
OFF_SPEC_HORIZON = 30
TRANSITION_BLOCK_SIZE = 1800  # steady(600) + transition(600) + steady(600), verified against data


def despike(series, window=7, n_sigmas=5.0):
    """Hampel filter: replace points far from the local median relative to local MAD."""
    k = 1.4826  # scales MAD to a normal-equivalent sigma
    rolling_median = series.rolling(window, center=True, min_periods=1).median()
    residual = (series - rolling_median).abs()
    mad = residual.rolling(window, center=True, min_periods=1).median()
    threshold = (n_sigmas * k * mad).replace(0, np.nan)
    is_spike = (residual > threshold).fillna(False)
    return series.where(~is_spike, rolling_median)


def add_transition_id(df):
    n_transitions = df.index.max() // TRANSITION_BLOCK_SIZE + 1
    expected_rows = n_transitions * TRANSITION_BLOCK_SIZE
    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {n_transitions} transitions of "
            f"{TRANSITION_BLOCK_SIZE} rows each, got {len(df)}. transition_id "
            f"assignment assumes the standard simulator block layout."
        )
    df["transition_id"] = df.index // TRANSITION_BLOCK_SIZE
    return df


def add_signed_deviation(df):
    df["bw_error_pct"] = (
        (df["basis_weight"] - df["basis_weight_setpoint"]) / df["basis_weight_setpoint"] * 100
    )
    return df


def add_rolling_features(df):
    despiked = {var: despike(df[var]) for var in DESPIKE_VARS}
    for var in ROLLING_VARS:
        series = despiked.get(var, df[var])
        for window in ROLLING_WINDOWS:
            df[f"{var}_roll_mean_{window}"] = series.rolling(window, min_periods=1).mean()
            df[f"{var}_roll_std_{window}"] = series.rolling(window, min_periods=1).std()
    return df


def add_rate_of_change(df, dt=1.0):
    for var in MANIPULATED_VARS:
        df[f"{var}_roc"] = df[var].diff() / dt
    return df


def add_lag_features(df):
    for var in LAG_VARS:
        for offset in LAG_OFFSETS:
            df[f"{var}_lag{offset}"] = df[var].shift(offset)
    return df


def add_future_off_spec_label(df, horizon=OFF_SPEC_HORIZON):
    """Label at t = 1 if off_spec is ever 1 in (t, t+horizon]. Excludes the current step
    since the goal is to flag risk before it happens, not to report the present."""
    reversed_off_spec = df["off_spec"][::-1].reset_index(drop=True)
    forward_max = reversed_off_spec.rolling(horizon, min_periods=1).max()[::-1].reset_index(drop=True)
    df["off_spec_future"] = forward_max.shift(-1).values
    return df


def build_features(df):
    df = df.copy()
    df = add_transition_id(df)
    df = add_signed_deviation(df)
    df = add_rolling_features(df)
    df = add_rate_of_change(df)
    df = add_lag_features(df)
    df = add_future_off_spec_label(df)

    max_lag = max(LAG_OFFSETS)
    df = df.iloc[max_lag: len(df) - OFF_SPEC_HORIZON].reset_index(drop=True)
    df["off_spec_future"] = df["off_spec_future"].astype(int)
    return df


def main():
    df = pd.read_csv(RAW_PATH)
    features = build_features(df)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(features)} rows, {features.shape[1]} columns -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
