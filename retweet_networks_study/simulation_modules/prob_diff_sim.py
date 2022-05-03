"""
This script contains functions to generate probability difference coefficient data.
It does so by simulating homophily and comparing the probability of an ego retweeting a more extreme peer
in the homophily condition to the ego's empirical probability of retweeting a more extreme peer. It does so
in the following steps:

1. Run the simulation on each ego rating in a random 70% subset of the data.
2. Count whether a chosen peer is more extreme than the ego for each condition (homophily and empirical).
3. Store the results and repeat the simulation for 100 trial on a different random subset each trial.
4. For each ego, divide the count of more extreme peers by the count of peers overall for each ego in each condition
to show the probability of a chosen peer being more extreme.
5. Subtract the probability of a peer being more extreme in the homophily simulation from the empirical
probability of a peer being more extreme.
6. Store the final results in a dataframe.
7. Save the results as a CSV file.
"""

# Import libraries:
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

import warnings
warnings.filterwarnings("ignore")


# Read in data:
def load_twitter_data():

    print('Loading data.', flush=True)

    # Load users data frame:
    users_df = pd.read_csv('../data/users_ratings.csv')
    users_df = users_df.set_index('userid')

    # Retweet network df:
    # rt = pd.read_csv('../data/rt_network.csv')
    rt_df = pd.read_csv('../data/rt_network.csv')

    return users_df, rt_df


# Preprocess for simulation:
def preprocess_data(users_df, rt_df, orient, frac_data=False, frac_start=None, frac_end=None):

    print('Data loaded. Preprocessing data.', flush=True)

    # Set minimum number of tweets:
    min_tweets = 5

    # Subset to conservative ego ratings:
    if orient == 'right':
        users_df = users_df[users_df['orig_rating'] > 0]

    # Subset to liberal ego ratings and convert ratings to positive scale:
    elif orient == 'left':
        users_df = users_df[users_df['orig_rating'] < 0]
        users_df['orig_rating'] = users_df['orig_rating'] * -1

    # Subset based on min tweet threshold:
    users_df = users_df[users_df['orig_total_count'] >= min_tweets]

    # Subset retweet network ID to contain only egos and peers that meet min tweet threshold:
    rt_df = rt_df[rt_df['userid'].isin(users_df.index) & rt_df['rt_userid'].isin(users_df.index)]

    # Remove observations where user retweeted self
    rt_df = rt_df[rt_df['userid'] != rt_df['rt_userid']]

    # Subset fraction of users to speed up simulation:
    if frac_data == True:

        print('Subsetting data into specified fraction of users.', flush=True)

        # Assert conditions:
        assert isinstance(frac_start, float) and isinstance(frac_end, float), "Fractions of data must be float type"
        assert frac_start >= 0.0 and frac_end > frac_start, "Must be defined fraction of data"

        # Get unique user ID values:
        all_users = np.unique(rt_df['userid'].values)

        # Subset to specified fraction of users:
        users_fraction = all_users[int(frac_start * len(all_users)):int(len(all_users) * frac_end)]

        # Return dataset with only user IDs in specified fraction:
        rt_df = rt_df[rt_df['userid'].isin(users_fraction)]

    return rt_df


# Join user and rt network datasets:
def join_data(users_df, rt_df):

    print('Data preprocessed. Joining datasets.', flush=True)
    
    # Join on user ID and retweet user ID:
    rt_df = rt_df.join(users_df[['orig_rating']], on='userid').join(users_df[['orig_rating']],
                                                      on='rt_userid',
                                                      rsuffix='_peer').rename(
        columns={'orig_rating': 'orig_rating_ego'})

    return rt_df


# Run homophily simulation and append homophily peer ratings to new column in dataframe:
def get_homophily_ratings_df(rt_df, limit=0.7):
    # Randomize order of dataset for simulation trial:
    rt_df = rt_df.sample(frac=1)

    # Get all ego and peer ratings:
    ego_ratings = rt_df['orig_rating_ego'].values
    peer_ratings = rt_df['orig_rating_peer'].values

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
    rt_df= rt_df.iloc[0:len(closest_peers)]

    # Create new column for closest peers based on homophily strategy:
    rt_df['homoph_rating_peer'] = closest_peers

    return rt_df


# Creates dataframe with user id and probability differences after n trials:
def get_sim_df(users_df, rt_df, orient, n=100):

    # Vectorized lambda function to count whether peer rating is greater than ego rating:
    get_more_extreme_count = np.vectorize(lambda x, y: 1 if y > x else 0)

    # Join data:
    rt_df = join_data(users_df, rt_df)

    # Initialize dataframe that keeps track of whether a peer is more extreme in homophily and empirical conditions:
    sim_df = pd.DataFrame()

    if orient == 'right':
        print('Data joined. Beginning conservative group simulation.', flush=True)

    elif orient == 'left':
        print('Data joined. Beginning liberal group simulation.', flush=True)

    # Run for n trials and continually add to the more_extreme_count dataframe:
    for i in range(n):
        print(f'Current iteration: {i+1} of {n}', flush=True)

        # Get homophily ratings:
        rt_df_subset = get_homophily_ratings_df(rt_df)

        # Create columns counting whether peer is more extreme than ego in each condition:
        rt_df_subset['is_more_extreme_homoph'] = get_more_extreme_count(rt_df_subset['orig_rating_ego'],
                                                                   rt_df_subset['homoph_rating_peer'])
        rt_df_subset['is_more_extreme_empi'] = get_more_extreme_count(rt_df_subset['orig_rating_ego'],
                                                                 rt_df_subset['orig_rating_peer'])

        # Append results to dataframe:
        sim_df = sim_df.append(rt_df_subset, ignore_index=True)

        print('Simulation complete. Creating dataframe.', flush=True)

    return sim_df


def get_prob_diff_df(sim_df):
    # Gets the probability that a peer is more extreme in each condition for each ego:
    prob_diff_df = sim_df.groupby('userid', as_index=False).agg(prob_more_extreme_homoph=('is_more_extreme_homoph', 'mean'),
                                                                prob_more_extreme_empi=('is_more_extreme_empi', 'mean'))

    # Crates column of differences between probabilities in the empirical and homophily conditions:
    prob_diff_df['prob_diff'] = prob_diff_df['prob_more_extreme_empi'] - prob_diff_df['prob_more_extreme_homoph']

    print('Dataframe created. Saving to csv.', flush=True)

    return prob_diff_df[['userid', 'prob_diff']]


# Function to save dataframe:
def save_prob_diff_df(df, orient, frac_start, frac_end):

    assert orient == 'right' or orient == 'left', "political orientation must be right or left"

    if orient == 'right':
        path_beginning = '../data/prob_diff_cons_'

    elif orient == 'left':
        path_beginning = '../data/prob_diff_libs_'

    file_path = path_beginning+str(frac_start)+'_'+str(frac_end)+'.csv'
    df.to_csv(file_path, index=False)

    print('Dataframe saved.', flush=True)


# Function to run entire simulation and save results:
def run_sim(orient, frac_data=False, frac_start=0.0, frac_end=0.1):

    # Run full simulation:
    users_df, rt_df = load_twitter_data()
    rt_df_frac = preprocess_data(users_df, rt_df, orient, frac_data=frac_data, frac_start=frac_start, frac_end=frac_end)
    sim_df = get_sim_df(users_df, rt_df_frac, orient)
    prob_diff_df = get_prob_diff_df(sim_df)
    save_prob_diff_df(prob_diff_df, orient, frac_start, frac_end)

# Running simulation for both liberals and conservatives on full datasets:
if __name__ == '__main__':
    orient = 'left'
    run_sim(orient)

    orient = 'right'
    run_sim(orient, frac_data)
