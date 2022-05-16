import ast
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import ast

# Add command line arguments:
parser = ArgumentParser(prog='Acrophily Simulations')
parser.add_argument('-s', '--sim_type', default="acrophily",
                    help="Type of simulation you wish to merge data for (acrophily/prob_diff/mean_abs_diff)")
parser.add_argument('-p', '--poli_affil', default="left",
                    help="User political affiliation you wish to merge data for (left/right)")


class SimDataProcessor:

    def __init__(self, sim_type, poli_affil):
        self.sim_type = sim_type
        self.poli_affil = poli_affil

        # Initializing merged dataframe and folder/file paths:
        self.df = None
        self.data_folder_path = 'data'
        self.sim_file_path = None

    # Gets proportion of egos with peers more extreme on average:
    def get_prob_more_extreme(self, peer_ratings):
        n_more_extreme = len(self.agg_threshold_df[peer_ratings > self.agg_threshold_df['orig_rating_ego']])
        n_total = len(self.agg_threshold_df)
        prob_more_extreme = n_more_extreme / n_total

        return prob_more_extreme

    # Gets proportion confidence interval for probability of peer being more extreme:
    def get_proportion_confint(self, peer_rating_col):
        n_more_extreme = len(self.agg_threshold_df[peer_rating_col > self.agg_threshold_df['orig_rating_ego']])
        n_total = len(self.agg_threshold_df)
        confint = proportion_confint(n_more_extreme, n_total)

        return confint

    @staticmethod
    def get_ci_upper_lower_cols(df, sim_type):

        df['confint_empi_lower'] = df['confint_empi'].apply(lambda x: x[0])
        df['confint_empi_upper'] = df['confint_empi'].apply(lambda x: x[1])

        if sim_type == 'mean_abs_diff':
            df['confint_random_lower'] = df['confint_random'].apply(lambda x: x[0])
            df['confint_random_upper'] = df['confint_random'].apply(lambda x: x[1])
        else:
            df['confint_homoph_lower'] = df['confint_homoph'].apply(lambda x: x[0])
            df['confint_homoph_upper'] = df['confint_homoph'].apply(lambda x: x[1])

            df['confint_acroph_lower'] = df['confint_acroph'].apply(lambda x: x[0])
            df['confint_acroph_upper'] = df['confint_acroph'].apply(lambda x: x[1])

        return df

    # Get file path based on simulation:
    def get_sim_file_path(self):
        if self.sim_type == 'prob_diff':
            sim_file_name = f'user_coef_{self.poli_affil}.csv'
        else:
            sim_file_name = f'{self.sim_type}_sim_{self.poli_affil}.csv'

        self.sim_file_path = os.path.join(self.data_folder_path, sim_file_name)

    # Get list of data files in file path for simulation:
    def get_data_files(self):
        if os.path.exists(self.sim_file_path):
            raise Exception("File already exists. Will not overwrite.")

        else:
            self.df = pd.DataFrame()

            files = os.listdir(self.data_folder_path)
            sim_data_files = [os.path.join(self.data_folder_path, file) for file in files
                              if file.startswith(f'{self.sim_type}_sim_{self.poli_affil}_')]

            print(f'{len(sim_data_files)} files found for sim type {self.sim_type}. Merging files.',
                  flush=True)

        return sim_data_files

    def get_agg_sim_df(self):
        self.agg_sim_df = self.sim_df.groupby('threshold', as_index=False).agg('mean')

    def process_sim_files(self):
        self.agg_sim_df = get_ci_upper_lower_cols(self.agg_sim_df, self.sim_type)


    def merge_sim_files(self):

        # Get sim file path:
        self.get_sim_file_path()

        # Get sim data files:
        sim_data_files = self.get_data_files()

        # Iterate and append through files to merge:
        for file in sim_data_files:
            if self.sim_type == 'mean_abs_diff':
                current_df = pd.read_csv(file, converters={'confint_empi': ast.literal_eval,
                                                           'confint_random': ast.literal_eval})
            elif self.sim_type == 'acrophily':
                current_df = pd.read_csv(file, converters={'confint_empi': ast.literal_eval,
                                                           'confint_homoph': ast.literal_eval,
                                                           'confint_acroph': ast.literal_eval})
                current_df['threshold'] = range(1, 41)

            current_df = self.get_ci_upper_lower_cols(current_df, self.sim_type)

            self.df = pd.concat([self.df, current_df], axis=0, ignore_index=True)

        if self.sim_type == 'prob_diff':
            print('Files merged. Standardizing coefficients.', flush=True)
        else:
            print('Files merged. Averaging results.', flush=True)

    def process_sim_files(self):

        if self.sim_type == 'prob_diff':
            self.df['coef'] = self.df['prob_diff'] / np.std(self.df['prob_diff'].values)
            self.df = self.df[['userid', 'coef', 'poli_affil']]
            print('Coefficients standardized. Dataset processed. Saving merged file.', flush=True)
        else:
            self.df = self.df.groupby('threshold', as_index=False).agg('mean')
            self.df['poli_affil'] = np.repeat(self.poli_affil, len(self.df))
            print('Results averaged. Saving merged file.', flush=True)

    def save_merged_files(self):

        if not os.path.exists(self.sim_file_path):
            self.df.to_csv(self.sim_file_path, index=False)
            print('Merged file saved.', flush=True)
        else:
            raise Exception("File already exists. Will not overwrite it.")

    def run(self):
        self.merge_sim_files()
        self.process_sim_files()
        self.save_merged_files()


def main(args=None):
    args = parser.parse_args(args=args)

    sim_type = args.sim_type
    poli_affil = args.poli_affil
    data_processor = SimDataProcessor(sim_type=sim_type, poli_affil=poli_affil)
    data_processor.run()


if __name__ == '__main__':
    main()
