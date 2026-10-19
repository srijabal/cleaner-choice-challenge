"""Audit the saved tournament's household separation and selection evidence."""
import unittest

from experiments import read_run
from model_tournament import records_from_probabilities


class TournamentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = read_run(41)
        cls.folds = read_run(2)['outputs']['household_folds']

    def test_budget_and_prior_expectations(self):
        self.assertEqual(len(self.saved['inputs']['configurations']), 17)
        self.assertEqual(len(self.saved['outputs']['trial_ids']), 85)
        for trial_id in self.saved['outputs']['trial_ids']:
            trial = read_run(trial_id)
            self.assertEqual(trial['inputs']['parent_run'], 41)
            self.assertTrue(trial['expectation'])
            self.assertLess(trial['created_at'], trial['finished_at'])
            held_out = {hid for hid, fold in self.folds.items() if fold == trial['inputs']['outer_fold']}
            self.assertFalse(held_out & set(trial['inputs']['inner_household_folds']))

    def test_every_household_visit_scored_once_per_comparator(self):
        reference = read_run(7)['outputs']['predictions']['utility_current']
        expected = {r['visit_id'] for r in reference}
        for records in self.saved['outputs']['predictions'].values():
            self.assertEqual(len(records), 1200)
            self.assertEqual({r['visit_id'] for r in records}, expected)
            for row in records:
                self.assertAlmostEqual(sum(row['probabilities'].values()), 1)

    def test_nested_search_uses_recorded_selected_configuration(self):
        for outer in self.saved['outputs']['outer_details']:
            self.assertEqual(outer['selected_config'], outer['family_choices']['utility'])
        self.assertEqual(self.saved['outputs']['predictions']['nested_search'],
                         self.saved['outputs']['predictions']['utility'])
        self.assertFalse(self.saved['outputs']['prototype_changed'])

    def test_invalid_probabilities_fail_before_reporting_metrics(self):
        row = {'household': {}, 'visit': {}, 'history': []}
        with self.assertRaises(ValueError):
            records_from_probabilities([row], [[float('nan'), 0, 0, 0, 1]])


if __name__ == '__main__':
    unittest.main()
