"""Paper machine simulator

Refactor of Paper_Machine_Simulator.ipynb's PaperMachineSimulator class.
Behavior is unchanged from the notebook; the only structural change is that
the recipe table is no longer hardcoded here -- it's loaded from
data/recipes/grade_recipes.json, the single source of truth also read by
models/setpoint_recommender.py for its safe-limit checks.
"""

import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

RECIPES_PATH = Path("data/recipes/grade_recipes.json")


def load_recipes(path=RECIPES_PATH):
    recipes = json.load(open(path))
    meta = recipes.pop("_meta")
    return recipes, meta


class PaperMachineSimulator:
    def __init__(self, dt=1.0, recipes_path=RECIPES_PATH):
        self.dt = dt
        self.time = 0.0

        recipes, meta = load_recipes(recipes_path)
        self.recipes = recipes
        self.width = meta["width"]
        self.retention = 0.75
        self.consistency = meta["consistency"]

        self.current_grade = "Grade_A"
        self.target_grade = "Grade_A"
        self.is_transitioning = False
        self.transition_scenario = "Normal"

        self.stock_flow = 1200.0
        self.filler_flow = 100.0
        self.machine_speed = 800.0
        self.steam_pressure = 60.0

        self.basis_weight = 60.0
        self.moisture = 6.5
        self.ash_content = 10.0
        self.dryer_temp = 120.0

        self.basis_weight_sp = 60.0
        self.moisture_sp = 6.5
        self.ash_sp = 10.0

        self.distance_to_scanner = 100.0
        self.buffer_size = 1200
        self.stock_buf = deque([self.stock_flow] * self.buffer_size, maxlen=self.buffer_size)
        self.filler_buf = deque([self.filler_flow] * self.buffer_size, maxlen=self.buffer_size)
        self.speed_buf = deque([self.machine_speed] * self.buffer_size, maxlen=self.buffer_size)
        self.steam_buf = deque([self.steam_pressure] * self.buffer_size, maxlen=self.buffer_size)

        self.bw_calibration_drift = 0.0
        self.last_bw_cal_time = 0.0
        self.frozen_sensor_val = None
        self.frozen_sensor_duration = 0

        self.data_log = []

    def get_transport_delay_steps(self, speed):
        speed_mps = speed / 60.0
        if speed_mps == 0:
            return self.buffer_size
        delay_seconds = self.distance_to_scanner / speed_mps
        return int(delay_seconds / self.dt)

    def apply_sensor_noise_and_artifacts(self, value, nominal_value, var_name):
        noise_std = 0.0005 * nominal_value
        value += np.random.normal(0, noise_std)

        if random.random() < 0.001:
            value += np.random.normal(0, 5 * noise_std) * 10

        if var_name == "basis_weight":
            drift_rate_per_sec = 0.02 / 3600.0
            self.bw_calibration_drift += drift_rate_per_sec * self.dt
            value += self.bw_calibration_drift
            if (self.time - self.last_bw_cal_time) > (8 * 3600):
                self.bw_calibration_drift = 0.0
                self.last_bw_cal_time = self.time

        if self.frozen_sensor_duration > 0:
            self.frozen_sensor_duration -= 1
            return self.frozen_sensor_val
        elif random.random() < 0.00005:
            self.frozen_sensor_val = value
            self.frozen_sensor_duration = random.randint(1, 3)
            return self.frozen_sensor_val

        return value

    def apply_process_disturbances(self):
        if random.random() < 0.0001:
            self.consistency += random.uniform(-0.0008, 0.0008)
        if random.random() < 0.00005:
            self.retention = max(0.65, self.retention - 0.10)
        if random.random() < 0.00002:
            self.steam_pressure = max(30.0, self.steam_pressure - 8.0)

    def step(self, target_setpoints=None, transition_info=None):
        self.apply_process_disturbances()

        if target_setpoints:
            stock_tc, speed_tc, steam_tc, filler_tc = 15.0, 10.0, 90.0, 20.0

            if self.transition_scenario == "Slow Actuator":
                stock_tc *= 3
            elif self.transition_scenario == "Aggressive Operator":
                stock_tc /= 1.5
                speed_tc /= 1.5
                steam_tc /= 1.5

            self.stock_flow += (target_setpoints["stock_flow"] - self.stock_flow) * (self.dt / stock_tc)
            self.filler_flow += (target_setpoints["filler_flow"] - self.filler_flow) * (self.dt / filler_tc)
            self.machine_speed += (target_setpoints["machine_speed"] - self.machine_speed) * (self.dt / speed_tc)

            steam_target = target_setpoints["steam_pressure"]
            if self.transition_scenario == "Steam Lag":
                delayed_steam_target = self.steam_buf[max(0, len(self.steam_buf) - int(45 / self.dt))]
                self.steam_pressure += (delayed_steam_target - self.steam_pressure) * (self.dt / steam_tc)
            else:
                self.steam_pressure += (steam_target - self.steam_pressure) * (self.dt / steam_tc)

            self.basis_weight_sp = target_setpoints["basis_weight_sp"]
            self.moisture_sp = target_setpoints["moisture_sp"]
            self.ash_sp = target_setpoints["ash_sp"]

        self.stock_buf.append(self.stock_flow)
        self.filler_buf.append(self.filler_flow)
        self.speed_buf.append(self.machine_speed)
        self.steam_buf.append(self.steam_pressure)

        delay_steps = self.get_transport_delay_steps(self.machine_speed)
        delayed_stock = self.stock_buf[max(0, len(self.stock_buf) - delay_steps)]
        delayed_speed = self.speed_buf[max(0, len(self.speed_buf) - delay_steps)]
        delayed_filler = self.filler_buf[max(0, len(self.filler_buf) - delay_steps)]
        delayed_steam = self.steam_buf[max(0, len(self.steam_buf) - delay_steps)]

        if delayed_speed > 0:
            bw_raw = (delayed_stock * self.consistency * self.retention * 1000) / (delayed_speed * self.width)
            self.basis_weight += (bw_raw - self.basis_weight) * (self.dt / 15.0)

        residence_time_factor = 1.0 / (delayed_speed / 60.0)
        steam_effect = delayed_steam / 20.0
        moisture_target = 5.0 + (residence_time_factor * 0.1) - steam_effect
        self.moisture += (moisture_target - self.moisture) * (self.dt / 90.0)

        if delayed_stock > 0:
            ash_target = (delayed_filler / delayed_stock) * 100.0
            self.ash_content += (ash_target - self.ash_content) * (self.dt / 30.0)

        temp_target = 100.0 + (self.steam_pressure * 0.8)
        self.dryer_temp += (temp_target - self.dryer_temp) * (self.dt / 5.0)

        nominal = self.recipes[self.current_grade] if self.current_grade in self.recipes else None
        bw_obs = self.apply_sensor_noise_and_artifacts(
            self.basis_weight, nominal["bw"] if nominal else self.basis_weight_sp, "basis_weight"
        )
        moisture_obs = self.apply_sensor_noise_and_artifacts(
            self.moisture, nominal["moisture"] if nominal else self.moisture_sp, "moisture"
        )
        ash_obs = self.apply_sensor_noise_and_artifacts(
            self.ash_content, nominal["ash"] if nominal else self.ash_sp, "ash_content"
        )

        deviation_percent = abs(bw_obs - self.basis_weight_sp) / self.basis_weight_sp * 100 if self.basis_weight_sp > 0 else 0
        off_spec = 1 if deviation_percent > self.recipes[self.current_grade]["bw_spec"] * 100 else 0

        self.data_log.append({
            "timestamp": self.time,
            "basis_weight": bw_obs,
            "stock_flow": self.stock_flow,
            "filler_flow": self.filler_flow,
            "machine_speed": self.machine_speed,
            "steam_pressure": self.steam_pressure,
            "moisture": moisture_obs,
            "ash_content": ash_obs,
            "dryer_temp": self.dryer_temp,
            "basis_weight_setpoint": self.basis_weight_sp,
            "moisture_setpoint": self.moisture_sp,
            "ash_setpoint": self.ash_sp,
            "deviation_percent": deviation_percent,
            "off_spec": off_spec,
            "grade": self.current_grade,
            "is_transitioning": transition_info["is_transitioning"] if transition_info else False,
            "transition_scenario": transition_info["scenario"] if transition_info else "SteadyState",
        })

        self.time += self.dt

    def run_transition(self, start_grade, end_grade, duration_steps=1200, scenario="Normal"):
        self.current_grade = start_grade
        self.target_grade = end_grade
        self.is_transitioning = True
        self.transition_scenario = scenario

        start_vals, end_vals = self.recipes[start_grade], self.recipes[end_grade]
        self.basis_weight_sp = start_vals["bw"]
        self.moisture_sp = start_vals["moisture"]
        self.ash_sp = start_vals["ash"]

        stock_flow_calc_end = (end_vals["bw"] * end_vals["speed"] * self.width) / (self.consistency * self.retention * 1000)
        stock_ramp_rate = (stock_flow_calc_end - self.stock_flow) / duration_steps
        speed_ramp_rate = (end_vals["speed"] - self.machine_speed) / duration_steps
        steam_ramp_rate = (end_vals["steam"] - self.steam_pressure) / duration_steps
        filler_ramp_rate = (end_vals["filler"] - self.filler_flow) / duration_steps

        bw_sp_ramp_rate = (end_vals["bw"] - self.basis_weight_sp) / duration_steps
        moisture_sp_ramp_rate = (end_vals["moisture"] - self.moisture_sp) / duration_steps
        ash_sp_ramp_rate = (end_vals["ash"] - self.ash_sp) / duration_steps

        for i in range(duration_steps):
            current_stock_target = self.stock_flow + stock_ramp_rate
            current_speed_target = self.machine_speed + speed_ramp_rate
            current_steam_target = self.steam_pressure + steam_ramp_rate
            current_filler_target = self.filler_flow + filler_ramp_rate

            current_bw_sp = self.basis_weight_sp + bw_sp_ramp_rate
            current_moisture_sp = self.moisture_sp + moisture_sp_ramp_rate
            current_ash_sp = self.ash_sp + ash_sp_ramp_rate

            if scenario == "Disturbed Trans." and i == int(duration_steps / 2):
                self.consistency += 0.0015
            elif scenario == "Recipe Mismatch":
                current_stock_target *= 1.02

            target_setpoints = {
                "stock_flow": current_stock_target,
                "filler_flow": current_filler_target,
                "machine_speed": current_speed_target,
                "steam_pressure": current_steam_target,
                "basis_weight_sp": current_bw_sp,
                "moisture_sp": current_moisture_sp,
                "ash_sp": current_ash_sp,
            }
            self.step(target_setpoints, {"is_transitioning": True, "scenario": scenario})

        self.is_transitioning = False
        self.current_grade = end_grade
        self.transition_scenario = "SteadyState"

    def run_steady_state(self, grade, duration_steps):
        self.current_grade = grade
        self.is_transitioning = False
        self.transition_scenario = "SteadyState"

        grade_vals = self.recipes[grade]
        self.basis_weight_sp = grade_vals["bw"]
        self.moisture_sp = grade_vals["moisture"]
        self.ash_sp = grade_vals["ash"]

        stock_flow_calc = (grade_vals["bw"] * grade_vals["speed"] * self.width) / (self.consistency * self.retention * 1000)

        target_setpoints = {
            "stock_flow": stock_flow_calc,
            "filler_flow": grade_vals["filler"],
            "machine_speed": grade_vals["speed"],
            "steam_pressure": grade_vals["steam"],
            "basis_weight_sp": grade_vals["bw"],
            "moisture_sp": grade_vals["moisture"],
            "ash_sp": grade_vals["ash"],
        }

        for _ in range(duration_steps):
            self.step(target_setpoints, {"is_transitioning": False, "scenario": "SteadyState"})

    def get_dataset(self):
        return pd.DataFrame(self.data_log)
