import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np

from sims_data_prep import TwitterDataProcessor

# Create probability difference simulation class that inherits processed data from TwitterDataProcessor:
class ProbDiffSim(TwitterDataProcessor):

    def __init__(self, orient, frac_data=False, frac_start=0.0, frac_end=0.1):
        TwitterDataProcessor.__init__(self, orient, frac_data, frac_start, frac_end)
        self.rt_df = self.get_retweet_data()

    # Run homophily simulation and append homophily peer ratings to new column in dataframe:
    def get_homophily_df(self, limit=0.7):

        # Randomize order of dataset for simulation trial:
        self.rt_df = self.rt_df.sample(frac=1)

        # Get all ego and peer ratings:
        ego_ratings = self.rt_df['orig_rating_ego'].values
        peer_ratings = self.rt_df['orig_rating_peer'].values

        # Initialize list of closest peers based on homophily simulation:
        closest_peers = []

        # For each ego rating in the first 70% of data, match ego with closest peer available based on min absolute difference:
        for ego_rating in ego_ratings[0:int(len(ego_ratings) * limit)]:
            # Find absolute differences:
            abs_diffs = np.abs(peer_ratings - ego_rating)

            # Find index of minimum absolute difference:
            min_diff_idx = abs_diffs.argmin()

            # Find closest peer at minimum difference index:
            closest_peer = peer_ratings[min_diff_idx]

            # Remove selected peer from peer pool:
            peer_ratings = np.delete(peer_ratings, min_diff_idx)

            # Append selected peer to closest peers list:
            closest_peers.append(closest_peer)

        # Create new dataset based on subset:
        self.homophily_df = self.rt_df.iloc[0:len(closest_peers)]

        # Create new column for closest peers based on homophily strategy:
        self.homophily_df['homoph_rating_peer'] = closest_peers


    # Creates dataframe with user id and probability differences after n trials:
    def get_sim_df(self, n=100):

        # Vectorized lambda function to count whether peer rating is greater than ego rating:
        get_more_extreme_count = np.vectorize(lambda x, y: 1 if y > x else 0)

        # Initialize dataframe that keeps track of whether a peer is more extreme in homophily and empirical conditions:
        self.sim_df = pd.DataFrame()

        if self.orient == 'right':
            print('Beginning conservative group simulation.', flush=True)

        elif self.orient == 'left':
            print('Beginning liberal group simulation.', flush=True)

        # Run for n trials and continually add to the more_extreme_count dataframe:
        for i in range(n):
            print(f'Current iteration: {i + 1} of {n}', flush=True)

            # Get homophily ratings:
            self.get_homophily_df()
            print(self.homophily_df.head(3))

            # Create columns counting whether peer is more extreme than ego in each condition:
            self.homophily_df['is_more_extreme_homoph'] = get_more_extreme_count(self.homophily_df['orig_rating_ego'],
                                                                            self.homophily_df['homoph_rating_peer'])
            self.homophily_df['is_more_extreme_empi'] = get_more_extreme_count(self.homophily_df['orig_rating_ego'],
                                                                          self.homophily_df['orig_rating_peer'])

            # Append results to dataframe:
            self.sim_df = pd.concat([self.sim_df, self.homophily_df], axis=0, ignore_index=True)

        print('Simulation complete. Creating dataframe.', flush=True)


    def get_prob_diff_df(self):
        # Get mean probability that a peer is more extreme in each condition for each ego across all trials:
        self.prob_diff_df = self.sim_df.groupby('userid', as_index=False).agg(
            prob_more_extreme_homoph=('is_more_extreme_homoph', 'mean'),
            prob_more_extreme_empi=('is_more_extreme_empi', 'mean'))

        # Crates column of differences between probabilities in the empirical and homophily conditions:
        self.prob_diff_df['prob_diff'] = self.prob_diff_df['prob_more_extreme_empi'] - self.prob_diff_df['prob_more_extreme_homoph']

        print('Dataframe created. Saving to csv.', flush=True)

    # Function to save dataframe:
    def save_prob_diff_df(self):

        if self.orient == 'right':
            path_beginning = '../data/prob_diff_cons_'

        elif self.orient == 'left':
            path_beginning = '../data/prob_diff_libs_'

        file_path = path_beginning + str(self.frac_start) + '_' + str(self.frac_end) + '.csv'
        self.prob_diff_df.to_csv(file_path, index=False)

        print('Dataframe saved.', flush=True)


    def run(self):
        self.get_sim_df()
        self.get_prob_diff_df()
        self.save_prob_diff_df()