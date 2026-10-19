"""Verify the saved split check and averaging do not inflate sample size."""
import unittest

from experiments import read_run
from split_robustness import average_losses


class SplitRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = read_run(127)
        cls.trials = [read_run(i) for i in cls.saved['outputs']['trial_ids']]

    def test_all_five_partitions_are_distinct_and_household_grouped(self):
        assignments = []
        for trial in self.trials:
            folds = trial['inputs']['household_folds']
            self.assertEqual(len(folds), 200)
            assignments.append(tuple(sorted(folds.items())))
            for fold in range(5):
                self.assertEqual(sum(v == fold for v in folds.values()), 40)
            for records in trial['outputs']['predictions'].values():
                self.assertEqual(len(records), 1200)
                self.assertEqual(len({r['visit_id'] for r in records}), 1200)
                self.assertEqual({r['household_id'] for r in records}, set(folds))
        self.assertEqual(len(set(assignments)), 5)

    def test_expectations_and_settings_are_recorded_before_results(self):
        self.assertEqual(self.saved['inputs']['seeds'], [101, 202, 303, 404, 505])
        for trial in self.trials:
            self.assertTrue(trial['expectation'])
            self.assertLess(trial['created_at'], trial['finished_at'])
            for fold in trial['outputs']['fold_models']:
                self.assertEqual(fold['models']['utility_current']['penalty'], 0.01)
                self.assertEqual(fold['models']['utility_current']['training_visits'], 960)
                self.assertEqual(fold['models']['logistic_current']['C'], 0.01)
        self.assertFalse(self.saved['outputs']['submitted_model_changed'])

    def test_repeated_losses_keep_1200_visits_and_200_households(self):
        averaged = average_losses([trial['outputs'] for trial in self.trials], 'utility_current')
        self.assertEqual(len(averaged), 1200)
        self.assertEqual(len({r['household_id'] for r in averaged}), 200)
        measured = sum(r['loss'] for r in averaged) / len(averaged)
        self.assertAlmostEqual(measured, self.saved['outputs']['overview']['utility_current']['log_loss']['mean'])
        for value in self.saved['outputs']['comparisons'].values():
            self.assertEqual(value['loss']['households'], 200)


if __name__ == '__main__':
    unittest.main()
