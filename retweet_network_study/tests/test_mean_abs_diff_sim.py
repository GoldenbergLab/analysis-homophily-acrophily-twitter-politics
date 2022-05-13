"""
This script tests functions from the mean abs diff simulation.
It tests that the dataframes are generated as expected,
with proper lengths and the correct number of columns.
"""

import os
import sys
import unittest
import numpy as np

# Find modules in proper directory:
appended_path = os.path.join('src', 'acrophily_sims')
sys.path.append(appended_path)

from acrophily_sims.acrophily_sims import MeanAbsDiffSim


class TestMeanAbsDiffSim(unittest.TestCase):

    # Set up class:
    @classmethod
    def setUpClass(cls):
        print('setupClass')

    # Tear down class:
    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    # Set up instances to test:
    def setUp(self):
        # Test for left, right, and fraction cases:
        self.sim_left = MeanAbsDiffSim(poli_affil='left',
                                       users_file=os.path.join('data', 'test_users.csv'),
                                       rt_file=os.path.join('data', 'test_rt.csv'),
                                       thresholds=range(1, 4))

        self.sim_right = MeanAbsDiffSim(poli_affil='right',
                                        users_file=os.path.join('data', 'test_users.csv'),
                                        rt_file=os.path.join('data', 'test_rt.csv'),
                                        thresholds=range(1, 4))

    # Tear down instances:
    def tearDown(self):
        print('tearDown')

    # Get both test cases for get_abs_diff_df function:
    def get_test_cases_abs_diff_df(self):
        self.sim_left.get_abs_diff_df()
        self.sim_right.get_abs_diff_df()

    # Test cases for get_sim_df function:
    def get_test_cases_sim_df(self):
        self.sim_left.get_sim_df()
        self.sim_right.get_sim_df()

    # Test cases for get_agg_sim_df function:
    def get_test_cases_agg_sim_df(self):
        self.sim_left.get_sim_df()
        self.sim_left.get_agg_sim_df()

        self.sim_right.get_sim_df()
        self.sim_right.get_agg_sim_df()

    # Assert df has proper number of columns for both cases:
    def assert_num_cols(self, df_name):

        # Gets column shape and asserts equal to expected amount for dataframe specified:
        if df_name == 'abs_diffs_df':
            n_cols_left = self.sim_left.abs_diff_df.shape[1]
            n_cols_right = self.sim_right.abs_diff_df.shape[1]

            self.assertEqual(n_cols_left, 7)
            self.assertEqual(n_cols_right, 7)

        elif df_name == 'threshold_df':
            n_cols_left = self.sim_left.threshold_df.shape[1]
            n_cols_right = self.sim_right.threshold_df.shape[1]

            self.assertEqual(n_cols_left, 7)
            self.assertEqual(n_cols_right, 7)

        elif df_name == 'agg_threshold_df':
            n_cols_left = self.sim_left.agg_threshold_df.shape[1]
            n_cols_right = self.sim_right.agg_threshold_df.shape[1]

            self.assertEqual(n_cols_left, 4)
            self.assertEqual(n_cols_right, 4)

        elif df_name == 'sim_df':
            n_cols_left = self.sim_left.sim_df.shape[1]
            n_cols_right = self.sim_right.sim_df.shape[1]

            self.assertEqual(n_cols_left, 4)
            self.assertEqual(n_cols_right, 4)

        elif df_name == 'agg_sim_df':
            n_cols_left = self.sim_left.agg_sim_df.shape[1]
            n_cols_right = self.sim_right.agg_sim_df.shape[1]

            self.assertEqual(n_cols_left, 6)
            self.assertEqual(n_cols_right, 6)

    # Assert proper df length for both test cases:
    def assert_proper_len(self, df_name):

        if df_name == 'abs_diff_df':

            # Get length of abs diff dfs:
            df_len_left = len(self.sim_left.abs_diff_df)
            df_len_right = len(self.sim_right.abs_diff_df)

            # Get rounded length of 70% of rt df:
            rt_df_frac_len_left = int(0.7 * len(self.sim_left.rt_df))
            rt_df_frac_len_right = int(0.7 * len(self.sim_right.rt_df))

            # Assert two lengths are almost equal for both test cases:
            self.assertAlmostEqual(df_len_left, rt_df_frac_len_left, delta=1)
            self.assertAlmostEqual(df_len_right, rt_df_frac_len_right, delta=1)

        # Assert aggregation of threshold df by unique user ID:
        elif df_name == 'agg_threshold_df':
            df_len_left = len(self.sim_left.agg_threshold_df)
            df_len_right = len(self.sim_right.agg_threshold_df)

            n_users_left = len(np.unique(self.sim_left.threshold_df['userid'].values))
            n_users_right = len(np.unique(self.sim_right.threshold_df['userid'].values))

            self.assertEqual(df_len_left, n_users_left)
            self.assertEqual(df_len_right, n_users_right)

        # Assert one unique row in final df for each threshold:
        elif df_name == 'agg_sim_df':
            df_len_left = len(self.sim_left.agg_sim_df)
            df_len_right = len(self.sim_right.agg_sim_df)

            n_thresholds_left = len(np.unique(self.sim_left.sim_df['threshold']))
            n_thresholds_right = len(np.unique(self.sim_right.sim_df['threshold']))

            self.assertEqual(df_len_left, n_thresholds_left)
            self.assertEqual(df_len_right, n_thresholds_right)

    def test_get_abs_diff_df(self):

        # Get test cases for tests:
        self.get_test_cases_abs_diff_df()

        # Assert correct number of cols:
        self.assert_num_cols(df_name='abs_diffs_df')

        # Assert proper df length:
        self.assert_proper_len(df_name='abs_diff_df')

    def test_get_sim_df(self):

        # Get test cases:
        self.get_test_cases_sim_df()

        # Assert correct number of cols for all dfs in function:
        self.assert_num_cols(df_name='abs_diff_df')
        self.assert_num_cols(df_name='threshold_df')
        self.assert_num_cols(df_name='agg_threshold_df')
        self.assert_num_cols(df_name='sim_df')

        # Assert proper lengths of relevant dfs:
        self.assert_proper_len(df_name='abs_diff_df')
        self.assert_proper_len(df_name='agg_threshold_df')

    def test_get_agg_sim_df(self):
        # Get test case:
        self.get_test_cases_agg_sim_df()

        # Assert proper number of columns:
        self.assert_num_cols(df_name='agg_sim_df')

        # Assert unique row for each threshold:
        self.assert_proper_len(df_name='agg_sim_df')


if __name__ == '__main__':
    unittest.main()
