"""
This script imports the 'run_sim' function from the simulation_modules directory
to run the probability difference simulation on a select fraction of either liberal
or conservative users in the dataset.

Arguments:
    orient: string, 'left' or 'right'
        Subsets users by political orientation
    frac_data: boolean, True or False
        Denotes whether to run simulation on fraction of users in dataset.
    frac_start: float, [0.0, 1.0)
        Decides on starting fraction of users if frac_data set to True.
    frac_end: float, (0.0, 1.0]
        Decides on end fraction of users if frac_data set to True.
"""

# Append path to import functions from prob diff sim module:
import sys
sys.path.append('../simulation_functions/')

# Import functions from prob diff sim module:
from prob_diff_sim_functions import run_sim

# Run for first 10% of liberal users in dataset:
if __name__ == '__main__':

    # Defining political orientation of users (left or right):
    orient = 'left'

    # Defining fraction of users to include in simulation:
    frac_data = True
    frac_start = 0.0
    frac_end = 0.03

    # Running simulation:
    run_sim(orient, frac_data=frac_data, frac_start=frac_start, frac_end=frac_end)
