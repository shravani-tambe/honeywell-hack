"""Closed-loop replay demo

Same grade-change scenario twice from an identical random seed --
once with the simulator's raw recipe ramp, once with
the recommendation engine watching the transition and nudging one
manipulated variable's ramp target whenever predicted off-spec risk is
elevated. Seed reused for both runs means disturbances, sensor spikes
and noise land at the same simulated instants in both runs, differences in 
stabilization attributed to recommendations, not randomness 

Usage (from the repo root): python closed_loop/replay_demo.py
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ("simulator", "features", "models"):
    sys.path.insert(0, str(ROOT / sub))

from paper_machine_simulator import PaperMachineSimulator  
import feature_engineering as fe  
import setpoint_recommender as recommender  

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUTPUT_DIR / "replay_results.json"
PLOT_PATH = OUTPUT_DIR / "trajectory_comparison.png"

SCENARIO = "Aggressive Operator"
START_GRADE, END_GRADE = "Grade_A", "Grade_C"
STEADY_STATE_DURATION = 600
TRANSITION_DURATION = 600
TAIL_DURATION = 600
ASSESS_INTERVAL = 30
STABILIZATION_HOLD = 30
SEED = 42


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)


def run_raw():
    seed_all()
    sim = PaperMachineSimulator()
    sim.run_steady_state(START_GRADE, STEADY_STATE_DURATION)
    transition_start = sim.time
    sim.run_transition(START_GRADE, END_GRADE, TRANSITION_DURATION, SCENARIO)
    sim.run_steady_state(END_GRADE, TAIL_DURATION)
    return sim.get_dataset(), transition_start


def transition_rates(sim, end_vals):
    stock_end = (end_vals["bw"] * end_vals["speed"] * sim.width) / (sim.consistency * sim.retention * 1000)
    return {
        "stock_flow": (stock_end - sim.stock_flow) / TRANSITION_DURATION,
        "machine_speed": (end_vals["speed"] - sim.machine_speed) / TRANSITION_DURATION,
        "steam_pressure": (end_vals["steam"] - sim.steam_pressure) / TRANSITION_DURATION,
        "filler_flow": (end_vals["filler"] - sim.filler_flow) / TRANSITION_DURATION,
        "bw_sp": (end_vals["bw"] - sim.basis_weight_sp) / TRANSITION_DURATION,
        "moisture_sp": (end_vals["moisture"] - sim.moisture_sp) / TRANSITION_DURATION,
        "ash_sp": (end_vals["ash"] - sim.ash_sp) / TRANSITION_DURATION,
    }, stock_end


def ramp_targets(sim, rates, step_index):
    targets = {
        "stock_flow": sim.stock_flow + rates["stock_flow"],
        "filler_flow": sim.filler_flow + rates["filler_flow"],
        "machine_speed": sim.machine_speed + rates["machine_speed"],
        "steam_pressure": sim.steam_pressure + rates["steam_pressure"],
        "basis_weight_sp": sim.basis_weight_sp + rates["bw_sp"],
        "moisture_sp": sim.moisture_sp + rates["moisture_sp"],
        "ash_sp": sim.ash_sp + rates["ash_sp"],
    }
    if SCENARIO == "Disturbed Trans." and step_index == TRANSITION_DURATION // 2:
        sim.consistency += 0.0015
    elif SCENARIO == "Recipe Mismatch":
        targets["stock_flow"] *= 1.02
    return targets


def build_current_row(sim):
    df = sim.get_dataset()
    df = fe.add_signed_deviation(df)
    df = fe.add_rolling_features(df)
    df = fe.add_rate_of_change(df)
    df = fe.add_lag_features(df)
    df["transition_id"] = 0
    return df.iloc[-1]


def assess(sim, model, feature_names, bounds, correlations):
    row = build_current_row(sim)
    baseline = float(
        model.predict_proba(recommender.build_feature_matrix(pd.DataFrame([row]), feature_names))[:, 1][0]
    )
    suggestion = None
    if recommender.RISK_BAND[0] <= baseline <= recommender.RISK_BAND[1]:
        suggestion = recommender.generate_suggestion(row, model, feature_names, bounds, correlations)
    return baseline, suggestion


def run_assisted(model, feature_names, bounds, correlations):
    seed_all()
    sim = PaperMachineSimulator()
    sim.run_steady_state(START_GRADE, STEADY_STATE_DURATION)
    transition_start = sim.time

    sim.current_grade, sim.target_grade = START_GRADE, END_GRADE
    sim.is_transitioning, sim.transition_scenario = True, SCENARIO
    end_vals = sim.recipes[END_GRADE]
    rates, stock_end = transition_rates(sim, end_vals)
    tail_targets = {
        "stock_flow": stock_end, "filler_flow": end_vals["filler"],
        "machine_speed": end_vals["speed"], "steam_pressure": end_vals["steam"],
        "basis_weight_sp": end_vals["bw"], "moisture_sp": end_vals["moisture"], "ash_sp": end_vals["ash"],
    }

    adjustment, applied, risk_log = None, [], []
    total_steps = TRANSITION_DURATION + TAIL_DURATION

    for i in range(total_steps):
        if i < TRANSITION_DURATION:
            targets = ramp_targets(sim, rates, i)
            transitioning, scenario_label = True, SCENARIO
        else:
            sim.current_grade, sim.is_transitioning, sim.transition_scenario = END_GRADE, False, "SteadyState"
            targets = dict(tail_targets)
            transitioning, scenario_label = False, "SteadyState"

        if adjustment:
            targets[adjustment["variable"]] += adjustment["delta"]

        sim.step(targets, {"is_transitioning": transitioning, "scenario": scenario_label})

        if i > 0 and i % ASSESS_INTERVAL == 0:
            baseline, suggestion = assess(sim, model, feature_names, bounds, correlations)
            risk_log.append({"timestamp": sim.time, "risk": baseline})
            adjustment = None
            if suggestion:
                suggestion["status"] = "auto_applied"
                adjustment = {
                    "variable": suggestion["variable"],
                    "delta": suggestion["recommended_value"] - suggestion["current_value"],
                }
                applied.append(suggestion)

    return sim.get_dataset(), transition_start, applied, risk_log


def stabilization_time(df, start_time, hold=STABILIZATION_HOLD):
    window = df[df["timestamp"] >= start_time].reset_index(drop=True)
    in_spec = window["off_spec"] == 0
    settled = (in_spec[::-1].rolling(hold, min_periods=hold).min()[::-1] == 1).fillna(False)
    settled_times = window.loc[settled, "timestamp"]
    return float(settled_times.iloc[0] - start_time) if not settled_times.empty else None


def off_spec_seconds(df, start_time):
    return float(df.loc[df["timestamp"] >= start_time, "off_spec"].sum())


def plot_comparison(raw_df, assisted_df, applied, risk_log, t0):
    context = 60
    raw_w = raw_df[raw_df["timestamp"] >= t0 - context]
    asst_w = assisted_df[assisted_df["timestamp"] >= t0 - context]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})

    sp = asst_w["basis_weight_setpoint"]
    ax1.fill_between(asst_w["timestamp"] - t0, sp * 0.975, sp * 1.025, color="0.85", label="\u00b12.5% spec band")
    ax1.plot(asst_w["timestamp"] - t0, sp, "--", color="0.4", linewidth=1, label="setpoint")
    ax1.plot(raw_w["timestamp"] - t0, raw_w["basis_weight"], color="#d62728", linewidth=1.1, label="raw (no assist)")
    ax1.plot(asst_w["timestamp"] - t0, asst_w["basis_weight"], color="#1f77b4", linewidth=1.1, label="assisted")
    ax1.set_ylabel("Basis Weight (GSM)")
    ax1.set_title(f"Grade change {START_GRADE}\u2192{END_GRADE} ({SCENARIO}): raw vs. assisted")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    if risk_log:
        risk_df = pd.DataFrame(risk_log)
        ax2.step(risk_df["timestamp"] - t0, risk_df["risk"], where="post", color="#555555", linewidth=1)
    ax2.axhspan(recommender.RISK_BAND[0], recommender.RISK_BAND[1], color="orange", alpha=0.12)
    for s in applied:
        ax2.axvline(s["timestamp"] - t0, color="#1f77b4", alpha=0.4, linestyle=":")
        ax2.annotate(s["variable"], (s["timestamp"] - t0, 0.95), rotation=90, fontsize=7, va="top")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("predicted off-spec\nrisk (assisted)")
    ax2.set_xlabel("seconds since transition start")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140)
    plt.close(fig)


def main():
    model = recommender.load_model()
    feature_names = list(model.feature_names_in_)
    recipes = recommender.load_recipes()
    bounds = recommender.recipe_bounds(recipes)
    correlations = recommender.load_correlations()

    raw_df, t0 = run_raw()
    assisted_df, t0_assisted, applied, risk_log = run_assisted(model, feature_names, bounds, correlations)
    assert t0 == t0_assisted

    raw_stab = stabilization_time(raw_df, t0)
    assisted_stab = stabilization_time(assisted_df, t0)
    raw_off = off_spec_seconds(raw_df, t0)
    assisted_off = off_spec_seconds(assisted_df, t0)

    results = {
        "scenario": SCENARIO,
        "start_grade": START_GRADE,
        "end_grade": END_GRADE,
        "seed": SEED,
        "transition_start_time": t0,
        "stabilization_time_seconds": {
            "raw": raw_stab,
            "assisted": assisted_stab,
            "improvement": (raw_stab - assisted_stab) if raw_stab is not None and assisted_stab is not None else None,
        },
        "off_spec_seconds_after_transition_start": {
            "raw": raw_off,
            "assisted": assisted_off,
            "reduction": raw_off - assisted_off,
        },
        "n_interventions_applied": len(applied),
        "interventions": applied,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    plot_comparison(raw_df, assisted_df, applied, risk_log, t0)

    print(f"stabilization time -- raw: {raw_stab}s  assisted: {assisted_stab}s")
    print(f"off-spec seconds   -- raw: {raw_off}  assisted: {assisted_off}")
    print(f"interventions applied: {len(applied)}")
    print(f"wrote {RESULTS_PATH} and {PLOT_PATH}")


if __name__ == "__main__":
    main()
