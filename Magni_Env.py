import gymnasium as gym
import numpy as np
from simglucose.simulation.env import T1DSimEnv
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.controller.base import Action
from custom_scenario_1 import Table1Scenario
from datetime import datetime


class DiabetesEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        patient_name="adult#010",
        insulin_cap=0.1,
        seed=1,
        days=5,
        verbose_log=False,
        print_every=20
    ):

        self.step_minutes = 3
        self.steps_per_day = 24 * 60 // self.step_minutes
        self.days = days
        self.episode_steps = self.days * self.steps_per_day

        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName("Dexcom", seed=seed)
        sensor.sample_time = self.step_minutes
        pump = InsulinPump.withName("Insulet")

        scenario = Table1Scenario(
            start_time=datetime(2025, 1, 1, 0, 0, 0),
            seed=seed,
            days=self.days
        )

        self._env = T1DSimEnv(patient, sensor, pump, scenario)
        self.insulin_cap = insulin_cap
        self.action_space = gym.spaces.Box(
            low=0.0, high=self.insulin_cap, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.iob_horizon_steps = 180 // self.step_minutes
        self.iob_kernel = np.array(
            [(self.iob_horizon_steps - k) / self.iob_horizon_steps
             for k in range(self.iob_horizon_steps)],
            dtype=np.float32
        )

        self.last_cgm = 0.0
        self.t = 0
        self.log_buffer = []
        self.verbose_log = verbose_log
        self.print_every = print_every


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        res = self._env.reset()

        if isinstance(res, tuple):
            raw_obs = res[0]
            info = res[1] if len(res) > 1 else {}
        else:
            raw_obs = res
            info = {}

        self.t = 0
        self.last_cgm = raw_obs[0]
        self.log_buffer = []
        self.insulin_history = np.zeros(self.episode_steps, dtype=np.float32)
        self.meal_array = self._build_meal_array()

        return self._get_obs(raw_obs), info


    def step(self, action):
        act_val = float(action.item()) if isinstance(action, np.ndarray) else float(action)
        self.insulin_history[self.t] = act_val * self.step_minutes

        action_obj = Action(basal=act_val, bolus=0.0)
        step_result = self._env.step(action_obj)

        if len(step_result) == 4:
            raw_obs, _, done, info = step_result
            truncated = False
        else:
            raw_obs, _, done, truncated, info = step_result

        cgm = raw_obs[0]
        reward = self._reward_magni(cgm)
        meal_raw = self.meal_array[self.t]
        meal_flag = 1 if meal_raw > 0 else 0

        self.log_buffer.append({
            "step": self.t,
            "time_min": self.t * self.step_minutes,
            "IU": act_val * self.step_minutes,
            "IOB": self._compute_iob(),
            "meal": meal_flag,
            "cgm": float(cgm)
        })

        if self.verbose_log:
            if (self.t % self.print_every == 0) or meal_flag == 1:
                print(
                    f"[t={self.t:4d} | {self.t*self.step_minutes:4d} min] "
                    f"CGM={cgm:6.1f} | "
                    f"IU={act_val*self.step_minutes:6.3f} | "
                    f"IOB={self._compute_iob():6.3f} | "
                    f"meal={meal_flag}"
                )

        obs = self._get_obs(raw_obs)

        self.last_cgm = cgm
        self.t += 1

        if self.t >= self.episode_steps:
            done = True
            truncated = True

        return obs, reward, done, truncated, info


    def _magni_risk(self, cgm: float) -> float:
        cgm = max(float(cgm), 1e-6)
        f = 1.509 * ((np.log(cgm) ** 1.084) - 5.381)
        return 10.0 * (f ** 2)

    def _reward_magni(self, cgm: float) -> float:
        cgm = float(cgm)
        if cgm < 70.0:
            return -1.0
        risk = self._magni_risk(cgm)
        risk_clip = np.clip(risk, 0.0, 15.5)
        reward = 1.0 - (risk_clip / 7.75)
        return float(reward)



    def _build_meal_array(self):
        meal_array = np.zeros(self.episode_steps, dtype=np.float32)
        start_time = self._env.scenario.start_time

        for meal_time, cho in self._env.scenario.scenario:
            delta_min = (meal_time - start_time).total_seconds() / 60
            step = int(delta_min // self.step_minutes)
            if 0 <= step < self.episode_steps:
                meal_array[step] += cho

        return meal_array

    def _compute_iob(self):
        start = max(0, self.t - self.iob_horizon_steps + 1)
        insulin_slice = self.insulin_history[start:self.t + 1][::-1]
        kernel_slice = self.iob_kernel[:len(insulin_slice)]
        return float(np.sum(insulin_slice * kernel_slice))

    def _get_obs(self, raw_obs):
        cgm = raw_obs[0]
        dcgm = cgm - self.last_cgm
        time_slot = ((self.t * self.step_minutes) // 120) % 12
        iob = self._compute_iob()

        meal_raw = self.meal_array[self.t]
        meal = 1.0 if meal_raw > 0 else 0.0

        return np.array(
            [cgm, dcgm, time_slot, iob, meal],
            dtype=np.float32
        )

    def render(self):
        return self._env.render()

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
