import ast
import os
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportion_confint


# Get file path based on simulation:
def get_sim_file_path(sim_type, poli_affil, data_folder_path='data'):
    if sim_type == 'prob_diff':
        sim_file_name = f'user_coef_{poli_affil}.csv'
    else:
        sim_file_name = f'{sim_type}_sim_{poli_affil}.csv'

    sim_file_path = os.path.join(data_folder_path, sim_file_name)

    return sim_file_path


def get_data_files(sim_type, poli_affil, sim_file_path, data_folder_path='data'):
    if os.path.exists(sim_file_path):
        raise Exception("File already exists. Will not overwrite.")

    if not sim_type == 'prob_diff':
        files = os.listdir(data_folder_path)
        sim_data_files = [os.path.join(data_folder_path, file) for file in files
                          if file.startswith(f'{sim_type}_sim_{poli_affil}_')]

        print(f'{len(sim_data_files)} files found for sim type {sim_type}. Merging files.',
              flush=True)
              
    else:
        files = os.listdir(data_folder_path)
        sim_data_files = [os.path.join(data_folder_path, file) for file in files
                        if file.startswith(f'{sim_type}_{poli_affil}_')]

        print(f'{len(sim_data_files)} files found for sim type {sim_type}. Merging files.',
            flush=True)

    return sim_data_files


def merge_sim_files(sim_type, poli_affil, data_folder_path='data'):
    # Get sim file path:
    sim_file_path = get_sim_file_path(sim_type, poli_affil, data_folder_path)

    # Get sim data files:
    sim_data_files = get_data_files(sim_type, poli_affil, sim_file_path)

    # Initialize dataframe:
    df = pd.DataFrame()

    # Iterate and append through files to merge:
    for file in sim_data_files:
        current_df = pd.read_csv(file)
        df = pd.concat([df, current_df], axis=0, ignore_index=True)

    print('Files merged. Processing dataframe.', flush=True)

    return df


# Gets proportion of egos with peers more extreme on average:
def get_probs_more_extreme(df, col):

    thresholds = np.unique(df['threshold'])

    probs_more_extreme = []

    for threshold in thresholds:
        threshold_df = df[df['threshold'] == threshold]
        n_more_extreme = len(threshold_df[threshold_df[col] > threshold_df['orig_rating_ego']])
        n_total = len(threshold_df)
        prob_more_extreme = n_more_extreme / n_total
        probs_more_extreme.append(prob_more_extreme)

    return probs_more_extreme


def get_probs_more_extreme_cols(df, agg_df):
    probs_more_extreme_empi = get_probs_more_extreme(df, col='orig_rating_peer')
    probs_more_extreme_homoph = get_probs_more_extreme(df, col='homoph_rating_peer')
    probs_more_extreme_acroph_min = get_probs_more_extreme(df, col='acroph_rating_peer_min')
    probs_more_extreme_acroph_max = get_probs_more_extreme(df, col='acroph_rating_peer_max')

    agg_df['prob_more_extreme_empi'] = probs_more_extreme_empi
    agg_df['prob_more_extreme_homoph'] = probs_more_extreme_homoph
    agg_df['prob_more_extreme_acroph_min'] = probs_more_extreme_acroph_min
    agg_df['prob_more_extreme_acroph_max'] = probs_more_extreme_acroph_max

    return agg_df


def get_proportion_confints(df, col):

    thresholds = np.unique(df['threshold'])

    confints_lower = []
    confints_upper = []

    for threshold in thresholds:
        threshold_df = df[df['threshold'] == threshold]
        n_more_extreme = len(threshold_df[threshold_df[col] > threshold_df['orig_rating_ego']])
        n_total = len(threshold_df)
        confint_lower, confint_upper = proportion_confint(n_more_extreme, n_total)

        confints_lower.append(confint_lower)
        confints_upper.append(confint_upper)

    return confints_lower, confints_upper


def get_confints(df, col):

    thresholds = np.unique(df['threshold'])

    confints_lower = []
    confints_upper = []

    for threshold in thresholds:
        threshold_df = df[df['threshold'] == threshold]

        confint_lower = sms.DescrStatsW(threshold_df[col].values).tconfint_mean()[0]
        confint_upper = sms.DescrStatsW(threshold_df[col].values).tconfint_mean()[1]

        confints_lower.append(confint_lower)
        confints_upper.append(confint_upper)

    return confints_lower, confints_upper


def get_confint_cols(sim_type, df, agg_df):

    if sim_type == 'mean_abs_diff':
        confints_lower_empi, confints_upper_empi = get_confints(df, col='mean_abs_diff_empi')
        confints_lower_random, confints_upper_random = get_confints(df, col='mean_abs_diff_random')

        agg_df['ci_empi_lower'] = confints_lower_empi
        agg_df['ci_empi_upper'] = confints_upper_empi

        agg_df['ci_random_lower'] = confints_lower_random
        agg_df['ci_random_upper'] = confints_upper_random
    else:
        confints_lower_empi, confints_upper_empi = get_proportion_confints(df, col='orig_rating_peer')
        confints_lower_homoph, confints_upper_homoph = get_proportion_confints(df, col='homoph_rating_peer')
        confints_lower_acroph_min, confints_upper_acroph_min = get_proportion_confints(df, col='acroph_rating_peer_min')
        confints_lower_acroph_max, confints_upper_acroph_max = get_proportion_confints(df, col='acroph_rating_peer_max')

        agg_df['ci_empi_lower'] = confints_lower_empi
        agg_df['ci_empi_upper'] = confints_upper_empi

        agg_df['ci_homoph_lower'] = confints_lower_homoph
        agg_df['ci_homoph_upper'] = confints_upper_homoph

        agg_df['ci_acroph_lower_min'] = confints_lower_acroph_min
        agg_df['ci_acroph_upper_min'] = confints_upper_acroph_min
        
        agg_df['ci_acroph_lower_max'] = confints_lower_acroph_max
        agg_df['ci_acroph_upper_max'] = confints_upper_acroph_max

    return agg_df


def get_agg_df(sim_type, df, poli_affil):

    agg_df = df.groupby('threshold', as_index=False).agg('mean')

    if sim_type == 'acrophily':
        agg_df = get_probs_more_extreme_cols(df, agg_df)
        agg_df = get_confint_cols(sim_type, df, agg_df)

    if sim_type == 'mean_abs_diff':
        agg_df = get_confint_cols(sim_type=sim_type, df=df, agg_df=agg_df)
        agg_df.drop(['userid'], axis=1, inplace=True)

    agg_df['poli_affil'] = np.repeat(poli_affil, len(agg_df))

    print('Dataframe processed. Saving to CSV.', flush=True)

    return agg_df

def save_agg_df(sim_file_path, agg_df):

    if not os.path.exists(sim_file_path):
        agg_df.to_csv(sim_file_path, index=False)
        print('Processed dataframe successfully saved.', flush=True)
    else:
        raise Exception("File already exists. Will not overwrite it.")


if __name__ == '__main__':
    sim_type = 'acrophily'
    poli_affil = 'right'
    sim_file_path = get_sim_file_path(sim_type, poli_affil)
    df = merge_sim_files(sim_type, poli_affil)
    
    if not sim_type == 'prob_diff':
        agg_df = get_agg_df(sim_type, df, poli_affil)
        save_agg_df(sim_file_path, agg_df)
        
    else:
        save_agg_df(sim_file_path, df)
