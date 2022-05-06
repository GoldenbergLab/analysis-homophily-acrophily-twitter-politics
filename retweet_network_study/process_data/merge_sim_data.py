import os
import sys

# Append path to search for module:
appended_path = os.path.join('..', 'simulation_modules')
sys.path.append(appended_path)

# Import functions from prob diff sim module:
from sims_data_prep import SimDataProcessor

if __name__ == '__main__':
    sim_type = 'prob_diff'
    data_merger = SimDataProcessor(sim_type=sim_type)
    data_merger.run()
