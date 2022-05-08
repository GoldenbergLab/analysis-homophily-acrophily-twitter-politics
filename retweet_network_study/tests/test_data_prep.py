import os
import sys
import unittest
import numpy as np

# Find modules in proper directory:
appended_path = os.path.join('..', 'src', 'sims')
sys.path.append(appended_path)

from sims_data_prep import TwitterDataProcessor

# Change working directory to root:
os.chdir('..')

class TestTwitterDataProcessor(unittest.TestCase):

    # Set up class:
    @classmethod
    def setUp(cls):
        print('setupClass')

    # Tear down class:
    @classmethod
    def tearDown(cls):
        print('teardownClass')

    # Set up instances to test:
    def setUp(self):
        # Test for left, right, and fraction cases:
        self.data_prep_left = TwitterDataProcessor(orient='left')
        self.data_prep_right = TwitterDataProcessor(orient='right')
        self.data_prep_frac_left = TwitterDataProcessor(orient='left', frac_data=True,
                                                        frac_start=0.0, frac_end=0.5)

    # Tear down instances:
    def tearDown(self):
        print('tearDown')

    # Test loading raw data (all instances work same):
    def test_load_raw_data(self):
        # Tests that file exists:
        self.assertTrue(os.path.exists(self.data_prep_left.users_file_path))

    # Test output of processing and joining steps for three instances:
    def test_get_retweet_data(self):

        # Test preprocessing step for left-wing participants/full data:
        self.data_prep_left.load_raw_data()
        self.data_prep_left.preprocess_data()

        # Define left-wing user dataframe:
        left_users_df = self.data_prep_left.users_df

        # Assert that all user ratings were converted to positive scale:
        self.assertTrue(all(left_users_df['orig_rating'] > 0))

        # Test preprocessing step for right-wing/full data:
        self.data_prep_right.load_raw_data()
        self.data_prep_right.preprocess_data()

        # Define right-wing user dataframe and assert all positive ratings:
        right_users_df = self.data_prep_left.users_df
        self.assertTrue(all(right_users_df['orig_rating'] > 0))

        # Test preprocessing step for a fraction dataset:
        self.data_prep_frac_left.load_raw_data()
        self.data_prep_frac_left.preprocess_data()

        # Define retweet dataframe for fraction of left-wing users:
        frac_left_rt_df = self.data_prep_frac_left.rt_df

        # Get number of unique users in left-wing fraction dataframe:
        n_users_frac_left = len(np.unique(frac_left_rt_df['userid']))

        # Define fraction end of left-wing fraction dataframe:
        frac_left_end = self.data_prep_frac_left.frac_end

        # Get unique number of users in full left-wing retweet dataframe:
        left_rt_df = self.data_prep_left.rt_df
        n_users_left = len(np.unique(left_rt_df['userid']))

        # Assert that proper fraction of users represented in fraction dataframe:
        self.assertTrue(n_users_frac_left == int(frac_left_end * n_users_left))

        # Join left-wing dataframes:
        self.data_prep_left.join_data()

        # Assert that a non-zero length dataframe exists:
        self.assertTrue(len(self.data_prep_left.rt_df) > 0)

        # Assert non-zero length dataframe exists for fraction dataframe:
        self.data_prep_frac_left.join_data()
        self.assertTrue(len(self.data_prep_frac_left.rt_df) > 0)

if __name__ == '__main__':
    unittest.main()
