import os
import sys

appended_path = os.path.join('..', 'simulation_modules')
sys.path.append(appended_path)

# Import functions from prob diff sim module:
from sims_module import ProbDiffSim

if __name__ == '__main__':
    prob_diff_sim = ProbDiffSim(orient='right', frac_data=True,
                                frac_start=0.0, frac_end=0.01)
    prob_diff_sim.run()
