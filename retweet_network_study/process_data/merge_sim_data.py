import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# Change root directory:
os.chdir('..')

# Add command line arguments:
parser = ArgumentParser(prog='Acrophily Simulations')
parser.add_argument('-s', '--sim_type', default="acrophily", help="Type of simulation you wish to run (acrophily/prob_diff/mean_abs_diff/NA)")
parser.add_argument('-o', '--orient', default="left", help="User political orientation you wish to run simulation on (left/right)")

class SimDataProcessor:

    def __init__(self, sim_type, orient):
        self.sim_type = sim_type
        self.orient = orient

        # Initializing merged dataframe and folder/file paths:
        self.df = None
        self.data_folder_path = 'data'
        self.sim_file_path = None

        # Populating merged dataframe file and data folder path attributes:
        self.merge_sim_files()

    def merge_sim_files(self):

        if self.sim_type == 'prob_diff':
            sim_file_name = f'user_coef_{self.orient}.csv'
        else:
            sim_file_name = f'{self.sim_type}_sim_{self.orient}.csv'

        self.sim_file_path = os.path.join(self.data_folder_path, sim_file_name)

        if os.path.exists(self.sim_file_path):
            raise Exception("File already exists. Will not overwrite it.")

        else:
            self.df = pd.DataFrame()

            files = os.listdir(self.data_folder_path)
            sim_data_files = [os.path.join(self.data_folder_path, file) for file in files
                              if file.startswith(f'{self.sim_type}_{self.orient}')]

            print(f'{len(sim_data_files)} files found for sim type {self.sim_type}. Merging files.',
                  flush=True)

            for file in sim_data_files:
                current_df = pd.read_csv(file)

                self.df = pd.concat([self.df, current_df], axis=0, ignore_index=True)

            print('Files merged.', flush=True)

    def process_sim_files(self):

        if self.sim_type == 'prob_diff':
            self.df['coef'] = self.df['prob_diff'] / np.std(self.df['prob_diff'].values)
            self.df = self.df[['userid', 'coef', 'poli_affil']]

    def save_merged_files(self):

        print('Saving merged file.', flush=True)

        if not os.path.exists(self.sim_file_path):
            self.df.to_csv(self.sim_file_path, index=False)
            print('Merged file saved.', flush=True)
        else:
            raise Exception("File already exists. Will not overwrite it.")

    def run(self):
        self.process_sim_files()
        self.save_merged_files()


def main(args=None):
    args = parser.parse_args(args=args)

    sim_type = args.sim_type
    orient = args.orient
    data_processor = SimDataProcessor(sim_type=sim_type, orient=orient)
    data_processor.run()


if __name__ == '__main__':
    main()