"""
This script assumes that the dataframes generated from the probability difference simulation already exist,
and that the dataframes were created on 10% subsets of the users in the full dataset at a time for
each of conservative and liberal users (such that there are twenty files in total).
It then merges all users together for each group into a single dataframe and appends their group
affiliation as a column before generating a final coefficient by standardizing the probability differences
by dividing by the overall standard deviation across both groups. It finally saves the results as a CSV file.
"""

# Import libraries:
import pandas as pd
import numpy as np

# Create start and end ranges based on names of files:
start_range = np.arange(0.0, 1.0, 0.1)
end_range = np.arange(0.1, 1.1, 0.1)

# Initialize final dataframe:
df = pd.DataFrame()

# File path for current group:
group_file_path = '../data/prob_diff_'+group+'_'

# Enumerate through start and end ranges:
for idx, val in enumerate(end_range):
    start_val = np.round(start_range[idx], 1)
    end_val = np.round(val, 1)

    # Complete file name:
    file_name = str(start_val)+'_'+str(end_val)+'.csv'

    # Full file path:
    full_file_path = group_file_path+file_name

    # Convert current full path to dataframe:
    current_df = pd.read_csv(full_file_path)

    # Add group name as affiliation column in dataframe:
    current_df['affiliation'] = np.repeat(group, len(current_df))

    # Append current dataframe to final dataframe:
    df = df.append(current_df, ignore_index=True)

# Divide probability differences by overall standard deviation to create coefficient column:
df['coef'] = df['prob_diff'] / np.std(df['prob_diff'])

# Select userid, coefficient, and affiliation columns:
coef_df = df[['userid', 'coef', 'affiliation']]

# Create final path:
final_path = '../data/user_coef.csv'

# Save final results:
coef_df.to_csv(final_path, index=False)
