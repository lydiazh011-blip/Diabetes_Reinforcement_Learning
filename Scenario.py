import numpy as np
from datetime import timedelta, datetime
from simglucose.simulation.scenario import Scenario
from collections import namedtuple

MealAction = namedtuple('MealAction', ['meal'])


class Table1Scenario(Scenario):
    def __init__(self, start_time=None, seed=None, days=5):
        super().__init__(start_time=start_time)
        self.rng = np.random.default_rng(seed)
        self.days = days
        self.scenario = []

    def reset(self):
        self.scenario = self.get_scenario(self.start_time)

    def get_action(self, t):
        for meal_time, meal_size in self.scenario:
            if meal_time == t:
                return MealAction(meal=meal_size)
        return MealAction(meal=0.0)

    def get_scenario(self, start_time):
        meals_config = [
            ('Breakfast', 0.60, 8, 1.5, 34, 7.5),
            ('Lunch',     0.99, 13, 0.5, 104, 22.5),
            ('Snack1',    0.30, 17, 1.0, 12, 2.5),
            ('Dinner',    0.95, 21, 1.0, 80, 17.5),
            ('Snack2',    0.03, 24, 1.0, 12, 2.5),
        ]

        scenario = []
        current_date = start_time.date()

        for day in range(self.days):
            day_date = current_date + timedelta(days=day)

            for _, prob, mean_h, std_h, mean_cho, std_cho in meals_config:
                if self.rng.random() < prob:
                    time_h = self.rng.normal(mean_h, std_h)
                    minutes = int(round(time_h * 60))
                    remainder = minutes % 3
                    if remainder != 0:
                        minutes += (3 - remainder)
                    minutes = max(0, minutes)

                    meal_time = (
                        datetime.combine(day_date, datetime.min.time())
                        + timedelta(minutes=minutes)
                    )

                    cho = max(0.0, self.rng.normal(mean_cho, std_cho))
                    scenario.append((meal_time, cho))

        scenario.sort(key=lambda x: x[0])
        return scenario

