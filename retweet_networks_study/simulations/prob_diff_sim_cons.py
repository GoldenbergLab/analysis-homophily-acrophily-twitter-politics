"""
This script runs the probability difference simulation on a select fraction of liberal users.
"""

# Append path to import functions from prob diff sim module:
import sys
sys.path.append('../functions/')

# Import functions from prob diff sim module:
from prob_diff_sim import run_sim

if __name__ == '__main__':
    # Defining political orientation of users:
    orient = 'right'

    # Defining fraction of users to include in simulation:
    frac_data = True
    frac_start = 0.0
    frac_end = 0.1

    # Running simulation:
    run_sim(orient, frac_data=frac_data, frac_start=frac_start, frac_end=frac_end)
