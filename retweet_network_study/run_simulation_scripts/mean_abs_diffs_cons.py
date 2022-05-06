"""
This script uses the random baseline simulation from the acrophily simulation script
to compare the mean absolute difference between each ego and the ego's peers to the
empirically observed mean absolute differences. It does so in the following steps:

1.  For each trial in the simulation, take a random 70% subset of data
2. For each ego in the random 70% subset, find the absolute difference between the
ego's political rating and its peers' political rating and take the average
3. Store the results as a pickle file for both the random condition and the empirical condition
"""

# Importing libraries
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

import warnings
warnings.filterwarnings('ignore')

import scipy

import pickle

# Defining functions:

# Gets confidence interval margin of error for mean absolute difference:
def mean_ci_moe(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    return (h)

# Subsets data by a random fraction of user IDs:
def get_random(r2, frac=1):
    us = r2['userid'].unique()
    us_sample = np.random.choice(us, size=int(len(us) * frac), replace=False)

    return (r2[r2['userid'].isin(us_sample)])


# Finds mean absolute difference between peer and ego ratings for retweet network:
def get_mean_abs_diff(r, u, thresh, baseline):
    # Determines retweet network by minimum number of retweets threshold:
    r1 = r[r['rt'] >= thresh]

    # Joins RT network DF with user DF on user id and on rt user id:
    r2 = r1.join(u[['orig_rating']], on='userid').join(u[['orig_rating']], on='rt_userid',
                                                       rsuffix='_peer').rename(
        columns={'orig_rating': 'orig_rating_ego'})

    r2 = get_random(r2, 0.7)

    if baseline == 'rand':
        r2['orig_rating_peer'] = np.random.permutation(r2['orig_rating_peer'])

    for ego_id in np.unique(r2['userid']):
        ego_df = r2[r2['userid'] == ego_id]

        peer_ratings = ego_df['orig_rating_peer'].values
        ego_rating = np.unique(ego_df['orig_rating_ego'])

        abs_diffs = np.abs(peer_ratings - ego_rating)

        # Mean absolute difference of user within threshold:
        mean_abs_diffs = np.mean(abs_diffs)

        # Append to list to have mean absolute difference per user:
        mean_abs_diffs_list.append(mean_abs_diffs)

    # Gets confidence interval moe:
    abs_diffs_ci_moe = mean_ci_moe(mean_abs_diffs_list)

    return np.mean(mean_abs_diffs_list), abs_diffs_ci_moe

# Repeats get_mean_abs_diff function n times and returns mean results:
def repeat_mean_abs_diff(r, u, thresh, baseline, n):
    mean_abs_diffs = []
    abs_diffs_ci_moes = []

    for i in range(n):
        mean_abs_diff, abs_diffs_ci_moe = get_mean_abs_diff(r, u, thresh, baseline)

        mean_abs_diffs.append(mean_abs_diff)
        abs_diffs_ci_moes.append(abs_diffs_ci_moe)

    return np.mean(mean_abs_diffs), np.mean(abs_diffs_ci_moes)

# Reading in data:
print('reading in data', flush=True)

users = pd.read_csv('../data/users_ratings.csv')
users = users.set_index('userid')

rt = pd.read_csv('../data/rt_network.csv')

# Print statement checkpoint for grid:
print('data successfully read', flush=True)

# Getting conversative user data only:
orient = 'right'
min_tweets = 5

if orient == 'right':
    u = users[users['orig_rating'] > 0]

u = u[u['orig_total_count'] >= min_tweets]

r = rt[rt['userid'].isin(u.index) & rt['rt_userid'].isin(u.index)]
r = r[r['userid'] != r['rt_userid']]  # Removes observations where user retweeted self

# Initializing dictionary:
baseline = {
    'rand':{},
    'empi':{}
}

# Setting number of trials:
n = 100

# Setting retweet number start and end range:
range_start = 5
range_end = 45

# Print statement for grid:
print('beginning simulation', flush=True)

# Creating baseline dictionaries for each model at each minimum tweet threshold within range:
for thresh in range(range_start, range_end, 5):
    models = ['rand', 'empi']

    # Print progress for grid:
    print(f'current threshold: {thresh}', flush=True)

    # Get mean abs diff and confidence interval MOE for each model at each threshold:
    for b in models:
        print(f'current model: {b}', flush=True)
        mean_abs_diff, abs_diff_ci_moe = repeat_mean_abs_diff(r, u, thresh, b, n)
        baseline[b][thresh] = {'mean_absolute_difference': mean_abs_diff,
                              'abs_diff_ci_moe': abs_diff_ci_moe}

# Print for grid:
print('simulation complete', flush=True)

# Save pickle file in data folder:
with open('../data/right_mean_abs_diffs_'+str(range_start)+'_'+str(range_end)+'_.pickle', 'wb') as handle:
    pickle.dump(baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Print for grid:
print('file saved', flush=True)
