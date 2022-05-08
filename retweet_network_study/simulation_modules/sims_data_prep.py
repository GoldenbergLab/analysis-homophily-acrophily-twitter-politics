import os
import numpy as np
import pandas as pd


class TwitterDataProcessor:

    def __init__(self, orient, frac_data=False, frac_start=None, frac_end=None):
        self.orient = orient
        self.frac_data = frac_data
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.users_file_path = None
        self.rt_network_file_path = None
        self.users_df = None
        self.rt_df = None

        # Defining data paths:
        self.data_path = os.path.join('..', 'data')
        self.users_file_path = os.path.join(self.data_path, 'users_ratings.csv')
        self.rt_network_file_path = os.path.join(self.data_path, 'rt_network.csv')

    def load_raw_data(self):

        print('Loading unprocessed user rating and retweet network data.', flush=True)
        # Load user ratings dataframe if file path exists:
        if os.path.exists(self.users_file_path):
            self.users_df = pd.read_csv(self.users_file_path)
            self.users_df = self.users_df.set_index('userid')
        else:
            return 'Users ratings file does not exist.'

        # Define retweet network dataframe if file path exists:
        if os.path.exists(self.rt_network_file_path):
            self.rt_df = pd.read_csv(self.rt_network_file_path)

        print('Datasets loaded. Processing and joining datasets.', flush=True)

    def preprocess_data(self):
        # Set minimum number of tweets:
        min_tweets=5

        # Subset to conservative ego ratings:
        if self.orient == 'right':
            self.users_df = self.users_df[self.users_df['orig_rating'] > 0]

        # Subset to liberal ego ratings and convert ratings to positive scale:
        elif self.orient == 'left':
            self.users_df = self.users_df[self.users_df['orig_rating'] < 0]
            self.users_df['orig_rating'] = self.users_df['orig_rating'] * -1

        # Subset based on min tweet threshold:
        self.users_df = self.users_df[self.users_df['orig_total_count'] >= min_tweets]

        # Subset retweet network ID to contain only egos and peers that meet min tweet threshold:
        userid_condition = self.rt_df['userid'].isin(self.users_df.index)
        rt_userid_condition = self.rt_df['rt_userid'].isin(self.users_df.index)

        self.rt_df = self.rt_df[userid_condition & rt_userid_condition]

        # Remove observations where user retweeted self
        self.rt_df = self.rt_df[self.rt_df['userid'] != self.rt_df['rt_userid']]

        # Subset fraction of users to speed up simulation:
        if self.frac_data is True:

            # Get unique user ID values:
            all_users = np.unique(self.rt_df['userid'].values)

            # Subset to specified fraction of users:
            n_users_start = int(self.frac_start*len(all_users))
            n_users_end = int(len(all_users) * self.frac_end)
            users_fraction = all_users[n_users_start:n_users_end]

            # Return dataset with only user IDs in specified fraction:
            self.rt_df = self.rt_df[self.rt_df['userid'].isin(users_fraction)]

    def join_data(self):
        # Join on user ID and retweet user ID:
        self.rt_df = self.rt_df.join(self.users_df[['orig_rating']],
                                     on='userid').join(self.users_df[['orig_rating']],
                                                       on='rt_userid',
                                                       rsuffix='_peer')\
            .rename(columns={'orig_rating': 'orig_rating_ego'})

        print('Datasets joined. Data successfully loaded.', flush=True)

    def get_retweet_data(self):
        self.load_raw_data()
        self.preprocess_data()
        self.join_data()

        return self.rt_df


class SimDataProcessor:

    def __init__(self, sim_type, orient):
        self.sim_type = sim_type
        self.orient = orient

        # Initializing merged dataframe and folder/file paths:
        self.df = None
        self.data_folder_path = None
        self.sim_file_path = None

        # Populating merged dataframe file and data folder path attributes:
        self.merge_sim_files()

    def merge_sim_files(self):
        self.data_folder_path = os.path.join('..', 'data')

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