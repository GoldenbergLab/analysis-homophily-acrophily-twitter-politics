#!/usr/bin/env python
# coding: utf-8

"""
This script runs the main acrophily simulation to compare a random baseline, a homophily simulation,
and an acrophily simulation to the empirically observed retweet strategies of users in our dataset.
It does so at various minimum retweet thresholds, finding both the average peer ratings and the
probability of retweeting a more extreme peer within each condition. It stores these results in
both a pickle file and as dataframes.
"""

# Import libraries:
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

import scipy.stats
import scipy

import statsmodels.stats.proportion as prop
import pickle

# Returns margin of error for mean of data array:
def mean_confidence_interval(data, confidence=0.95):
    
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    return (h)

# Selects fraction of IDs are chosen at random to include in network DF subset:
def get_random(r2, frac=1):
    
    us = r2['userid'].unique()
    us_sample = np.random.choice(us, size=int(len(us)*frac), replace=False)
    
    return(r2[r2['userid'].isin(us_sample)])


# Simulates either homophily (direction = 'any') or one-sided homophily (direction = 'right')
# Selects fraction of users with minimal difference in user leanings
def get_closest(r_,direction='any'):
    
    e = r_['orig_rating_ego'].values
    p = r_['orig_rating_peer'].values
    limit = 0.7
    closest=[]
    
    # Create 70% subset with max homophily users based on minimal distance in either direction:
    if direction == 'any':
        
        egos = []
        # For every ego rating in 70% subset of ego ratings:
        for ev in e[0:int(len(e)*limit)]:
            
            absolute_val_array = np.abs(p - ev) # abs peer polit minus ego polit
            smallest_difference_index = absolute_val_array.argmin()
            
            closest_element = p[smallest_difference_index]
            
            p = np.delete(p,smallest_difference_index)
            closest.append(closest_element)
            egos.append(ev)
            
        count_greater = len(np.array(closest)[np.array(closest) >= np.array(egos)])
        prop_greater = count_greater / len(np.array(closest))

        r_ = r_.iloc[0:len(closest)] # only participants in closest
        r_['orig_rating_peer'] = closest # Creates new column for closest peer ratings
        
          
    if direction == 'right':
        egos = []
        for ev in e[0:int(len(e)*limit)]:
            diff = p - ev # Returns difference array
            
            # Returns index of minimum difference element, penalizing differences under 0:
            smallest_difference_index = np.where(diff >= 0, diff, diff**2 + 100).argmin()
            
            # Uses index of closest element to grab element:
            closest_element = p[smallest_difference_index]
            
            # Deletes element for next iteration:
            p = np.delete(p,smallest_difference_index)
            
            # Appends element to 'closest' array:
            closest.append(closest_element)
            egos.append(ev)
            
        count_greater = len(np.array(closest)[np.array(closest) >= np.array(egos)])
        prop_greater = count_greater / len(np.array(closest))
            
        # Returns network subset with closest peer ratings:
        r_ = r_.iloc[0:len(closest)]
        r_['orig_rating_peer'] = closest
        
    return(r_)


def get_furthest(r_):
    
    e = r_['orig_rating_ego'].values
    p = r_['orig_rating_peer'].values
    diff = p - e
    pos_diff = [x for x in diff if x >= 0]
    r_ = r_.iloc[0:len(pos_diff)]
    r_['orig_rating_peer'] = r_[0:len(pos_diff)]
    
    return(r_)

# Runs subsetting for either of the homophily strategies depending on the specified direction:
def baseline_biased(r_,furthest=False,direction='any'):
    
    if furthest == True:
        r_ = get_furthest(r_)
    else:
        r_ = get_closest(r_,direction)
        
    return(r_)


# Gets a random sample of peers:
def baseline_rand(r_,sim_limit):
    
    r_short = r_.iloc[:int((len(r_)*sim_limit))]
    
    # Need to remove this line if we want to keep user ID consistent:
    r_short['orig_rating_ego'] = r_['orig_rating_ego'].sample(frac=1).iloc[:int((len(r_)*sim_limit))].values
    r_short['orig_rating_peer'] = r_['orig_rating_peer'].sample(frac=1).iloc[:int((len(r_)*sim_limit))].values
    
    return(r_short)


def get_base_ratings(r,u,thresh=5,baseline='rand'):
    
    """
    Returns mean political leanings of egos and peers, as well as the conditional probability 
    of being connected to more politically extreme person, along with its CI.
    
    It passes the retweet network DF (r), the user DF (u), a minimum retweet treshold (thresh), and the
    baseline model, where baseline = 'rand', 'homo', 'homo_right', or 'empi' depending on the strategy.
    The baseline argument will determine whether to use the baseline_rand or baseline_biased function to subset
    the dataframe, and, if simulating homophily, the direction argument will determine whether it is 
    one-sided or two-sided homophily. The thresh argument will determine the minimum number of retweets at which
    to subset the retweet network DF.
    
    """
    
    # Subsets based on min retweet threshold:
    r1 = r[r['rt']>=thresh]
    
    # Joins subset RT network DF with user DF based on user id and on rt user id:
    r2 = r1.join(u[['orig_rating']], on='userid').join(u[['orig_rating']], on='rt_userid', 
                                   rsuffix = '_peer').rename(columns={'orig_rating':'orig_rating_ego'})
    
    # If statement lines are to perform proper baseline sampling procedure depending on strategy:
    if baseline == 'rand':
        #r2['orig_rating_peer'] = r2['orig_rating_peer'].sample(frac=1).values
        r2 = baseline_rand(r2, 0.7)
        
    if baseline == 'homo':
        r2 = baseline_biased(r2.sample(frac=1),furthest=False)
        
    if baseline == 'acro': # does this matter? - maybe not
        r2 = baseline_biased(get_random(r2, .1),furthest=True)
        
    if baseline == 'homo_right':    
        r2 = baseline_biased(r2.sample(frac=1),direction='right')
    
    # Gets mean ego and peer ratings groupbed by user ID:
    ego_rating = r2.groupby('userid')['orig_rating_ego'].mean()
    peer_rating = r2.groupby('userid')['orig_rating_peer'].mean()
    
    # Creates new DF of users and original ratings:
    s = r2.drop_duplicates(subset=['userid'])[['userid','orig_rating_ego']].set_index('userid')
    
    # Joins with mean peer ratings grouped by user ID:
    s = s.join(r2.groupby('userid')['orig_rating_peer'].mean())
    
    # Number of ego ratings smaller than their peers' ratings:
    h = len(s[s['orig_rating_ego']<s['orig_rating_peer']])
    
    peer_prob_higher = h/(len(s))
    peer_prob_higher_ci = peer_prob_higher - prop.proportion_confint(h, len(s), alpha=0.05, method='normal')[0]
    
    return(ego_rating,peer_rating,peer_prob_higher,peer_prob_higher_ci)


# Function to repeat main function multiple times, returning mean probabilities of peers being higher w/mean CI:
def repeat_base_rating(r,u,baseline,direction='any',thresh=5,n=5):
    
    # Initializing series to append to for ego and peer ratings:
    ego_rating = pd.Series(dtype = 'float64')
    peer_rating = pd.Series(dtype = 'float64')
    
    peer_prob_higher = []
    peer_prob_higher_ci = []
    
    # Performs multiple iterations of simulation procedure:
    for i in range(n):
        eb,pb,pphb,pphc = get_base_ratings(r,u,thresh=thresh,baseline=baseline)
        
        # Concats results with series:
        ego_rating = pd.concat([ego_rating, eb])
        peer_rating = pd.concat([peer_rating, pb])
        
        peer_prob_higher.append(pphb)
        peer_prob_higher_ci.append(pphc)
        
    return(ego_rating,peer_rating,np.mean(peer_prob_higher),np.mean(peer_prob_higher_ci))

# Print statement to confirm script is running on grid:
print('reading in user data', flush=True)

# Read in data:
users = pd.read_csv('../data/users_ratings.csv')
users = users.set_index('userid')

# Get users with at least 5 original tweets:
u = users[users['orig_total_count']>=5]


# Read in retweet network df:
rt = pd.read_csv('../data/rt_network.csv')

# Print confirmation that data was read for grid:
print('data read', flush=True)


# Simulating retweet network for right-leaning individuals:
orient = 'right' # For conservatives
min_tweets = 5

if orient == 'right':
    u = users[users['orig_rating']>0] # Conservative user ratings are coded as above 0.

# Subset based on min tweet threshold:
u = u[u['orig_total_count']>=min_tweets]

# Subsetting retweet network ID to contain only egos and peers that meet min tweet threshold:
r_r = rt[rt['userid'].isin(u.index) & rt['rt_userid'].isin(u.index)]
np.random.seed(27)
r_r = r_r.sample(frac=0.5)

r_r = r_r[r_r['userid'] != r_r['rt_userid']] # Removes observations where user retweeted self

# Initializing dictionaries for each model:
baseline = {
    'rand':{},
    'homo':{},
    'homo_right':{},
    'empi':{}
}

# Print statement to check progress on grid:
print('beginning conservative simulation', flush=True)

# Creating minimum retweet threshold range:
range_start = 5
range_end = 45
n = 100

# Creating baseline dictionaries for each model at each minimum tweet threshold within range:
for thresh in range(range_start,range_end, 5):
    models = ['homo','homo_right', 'rand', 'empi']
    for b in models:
        # Print statement for grid:
        print('threshold = {0}, model = {1}'.format(thresh, b), flush=True)
        if b == 'homo':
            ego,peer,peer_higher, peer_higher_ci = repeat_base_rating(r_r,u,thresh=thresh,baseline='homo',n=n)
        if b == 'homo_right':
            ego,peer,peer_higher, peer_higher_ci = repeat_base_rating(r_r,u,thresh=thresh,baseline='homo_right',n=n)
        else:
            ego,peer,peer_higher, peer_higher_ci = get_base_ratings(r_r,u,thresh=thresh,baseline=b)
        baseline[b][thresh] = {'ego':ego.mean(), 'ego_ci':mean_confidence_interval(ego),
                                      'peer':peer.mean(),'peer_ci':mean_confidence_interval(peer),
                                      'peer_prob_higher':peer_higher, 'peer_prob_ci':peer_higher_ci}

# Print confirmation that simulation finished for grid:
print('conservative simulation complete', flush=True)

# Save results as pickle file:
with open('../data/baseline_right_'+str(range_start)+'_'+str(range_end)+'_.pickle', 'wb') as handle:
    pickle.dump(baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Print confirmation that data saved:
print('model data saved as pickle file', flush=True)

x = []
egos_base = []
egos_empi = []
peer_base = []
peer_base_h = []
peer_base_a = []
peer_empi = []

egos_base_ci = []
peer_base_ci = []
peer_base_h_ci = []
peer_base_a_ci = []
peer_empi_ci = []

pphb = [] # Probability peer higher baseline
pphbh = [] # Probability peer higher baseline homophily
pphba = [] # Probability peer higher baseline acrophily
pph = [] # Probability peer higher empirical

pphb_ci = []
pphbh_ci = []
pphba_ci = []
pph_ci = []

# For each min tweet threshold and within each model's dictionary, appends list of ego/peer values + ego/peer CIs:
for k,v in baseline['rand'].items():
    x.append(k)
    egos_base.append(v['ego'])
    egos_base_ci.append(v['ego_ci'])
 
    peer_base.append(v['peer'])
    peer_base_ci.append(v['peer_ci'])
    pphb.append(v['peer_prob_higher'])
    pphb_ci.append(v['peer_prob_ci'])

for k,v in baseline['homo'].items():
    peer_base_h.append(v['peer'])
    peer_base_h_ci.append(v['peer_ci'])
    pphbh.append(v['peer_prob_higher'])
    pphbh_ci.append(v['peer_prob_ci'])
    
for k,v in baseline['homo_right'].items():
    peer_base_a.append(v['peer'])
    peer_base_a_ci.append(v['peer_ci'])
    pphba.append(v['peer_prob_higher'])
    pphba_ci.append(v['peer_prob_ci'])
    
for k,v in baseline['empi'].items():
    egos_empi.append(v['ego'])
    peer_empi.append(v['peer'])
    peer_empi_ci.append(v['peer_ci'])
    pph.append(v['peer_prob_higher'])
    pph_ci.append(v['peer_prob_ci'])


# Creating csv file of expected peer political leanings with CIs for each model plus empirical at each threshold:
pd.DataFrame(list(zip(x, egos_base, peer_base, peer_base_ci, peer_base_h, peer_base_h_ci,
                      peer_base_a,peer_base_a_ci, egos_empi, peer_empi,peer_empi_ci)), 
               columns =['x', 'ego_baseline', 'peer_baseline', 'peer_baseline_ci',
                         'peer_homophily', 'peer_homophily_ci',
                         'peer_homophily_right', 'peer_homophily_right_ci',
                         'ego_empirical', 'peer_empirical', 'peer_empirical_ci']).to_csv('../data/leaning_right.csv',index=False)

# Creating csv file of probabilities at each threshold for each model plus empirical:
pd.DataFrame(list(zip(x, pphb, pphb_ci,pphbh,pphbh_ci,pphba,pphba_ci,pph,pph_ci)), 
               columns =['x', 'baseline', 'baseline_ci', 
                         'homophily', 'homophily_ci', 
                         'homophily_right', 'homophily_right_ci',
                         'empirical',
                         'empirical_ci']).to_csv('../data/right_acroph_sim_'+str(start_range)+'_'+str(end_range)+'.csv', index=False)

# Print confirmation that dataframes saved:
print('conservative dataframes saved', flush=True)



