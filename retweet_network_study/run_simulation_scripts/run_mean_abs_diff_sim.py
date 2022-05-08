import os
import sys

appended_path = os.path.join('..', 'simulation_modules')
sys.path.append(appended_path)

# Import functions from prob diff sim module:
from sims_module import MeanAbsDiffSim

if __name__ == '__main__':
    mean_abs_diff_sim = MeanAbsDiffSim(orient='right', thresholds=range(5, 45, 5))
    mean_abs_diff_sim.run()