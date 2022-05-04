# Append path to import functions from prob diff sim module:
import sys
sys.path.append('../simulation_modules/')

# Import functions from prob diff sim module:
from sims_module import ProbDiffSim

if __name__ == '__main__':
    prob_diff_sim = ProbDiffSim(orient='left', frac_data=True, frac_end=0.02)
    prob_diff_sim.run()