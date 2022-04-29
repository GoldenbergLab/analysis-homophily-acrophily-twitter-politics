"""
This script uses the homophily simulation from the acrophily simulation script and
compares the probability of retweeting a more extreme peer in the homophily condition
to the empirically observed data. It does so using the following steps:

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

# Vectorized lambda function to count whether peer rating is greater than ego rating:
get_more_extreme_count = np.vectorize(lambda x, y: 1 if y > x else 0)

# Preprocess data:
def preprocess_data(r):
    # Joins user dataframe with rt network dataframe:
    r2 = r.join(u[['orig_rating']], on='userid').join(u[['orig_rating']],
                                                      on='rt_userid',
                                                      rsuffix='_peer').rename(
        columns={'orig_rating': 'orig_rating_ego'})

    return r2

# Run homophily simulation and append homophily peer ratings to new column in dataframe:
def get_homophily_ratings(r2, limit=0.7):
    # Randomize order of dataset for simulation trial:
    r2 = r2.sample(frac=1)

    # Get all ego and peer ratings:
    ego_ratings = r2['orig_rating_ego'].values
    peer_ratings = r2['orig_rating_peer'].values

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
    r_short = r2.iloc[0:len(closest_peers)]

    # Create new column for closest peers based on homophily strategy:
    r_short['homoph_rating_peer'] = closest_peers

    return r_short


# Creates dataframe with user id and probability differences after n trials:
def get_sim_df(r, n=100):
    # Join data:
    r2 = preprocess_data(r)

    # Initialize dataframe that keeps track of whether a peer is more extreme in homophily and empirical conditions:
    more_extreme_count_df = pd.DataFrame()

    # Run for n trials and continually add to the more_extreme_count dataframe:
    for i in range(n):
        print(f'current iteration: {i+1} of {n}', flush=True)

        # Get homophily ratings:
        r_short = get_homophily_ratings(r2)

        # Create columns counting whether peer is more extreme than ego in each condition:
        r_short['is_more_extreme_homoph'] = get_more_extreme_count(r_short['orig_rating_ego'],
                                                                   r_short['homoph_rating_peer'])
        r_short['is_more_extreme_empi'] = get_more_extreme_count(r_short['orig_rating_ego'],
                                                                 r_short['orig_rating_peer'])

        # Append results to dataframe:
        more_extreme_count_df = more_extreme_count_df.append(r_short, ignore_index=True)

    # Gets the probability that a peer is more extreme in each condition for each ego:
    prob_diff_df = more_extreme_count_df.groupby('userid',
                                   as_index=False).agg(prob_more_extreme_homoph=('is_more_extreme_homoph', 'mean'),
                                                       prob_more_extreme_empi=('is_more_extreme_empi', 'mean'))


    # Crates column of differences between probabilities in the empirical and homophily conditions:
    prob_diff_df['prob_diff'] = prob_diff_df['prob_more_extreme_empi'] - prob_diff_df['prob_more_extreme_homoph']

    return prob_diff_df[['userid', 'prob_diff']]


print('reading in data', flush=True)
users = pd.read_csv('../data/users_ratings.csv')
users = users.set_index('userid')

# Get users w/ at least 5 written tweets
u = users[users['orig_total_count'] >= 5]

# Retweet network df:
# rt = pd.read_csv('../data/rt_network.csv')
rt = pd.read_csv('../data/rt_network.csv')

print('data successfully read', '\n', flush=True)

orient = 'right' # For conservatives
min_tweets = 5

if orient == 'right':
    u = users[users['orig_rating']>0] # Conservative user ratings are coded as above 0.

# Subset based on min tweet threshold:
u = u[u['orig_total_count']>=min_tweets]

# Subset retweet network ID to contain only egos and peers that meet min tweet threshold:
r = rt[rt['userid'].isin(u.index) & rt['rt_userid'].isin(u.index)]

# Remove observations where user retweeted self
r = r[r['userid'] != r['rt_userid']]

# Subset fraction of users to speed up simulation:
frac_start = 0.8
frac_end = 0.9

all_users = np.unique(r['userid'].values)
users_fraction = all_users[int(frac_start*len(all_users)):int(len(all_users)*frac_end)]

r_frac = r[r['userid'].isin(users_fraction)]

print('beginning conservative simulation', flush=True)

prob_diff_df = get_sim_df(r_frac)

file_path = '../data/prob_diff_cons_'+str(frac_start)+'_'+str(frac_end)+'.csv'
prob_diff_df.to_csv(file_path, index=False)

print('simulation complete. dataframe saved.', flush=True)
