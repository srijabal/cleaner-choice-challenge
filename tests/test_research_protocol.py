"""Audit the saved loop's separation of search and evaluation households."""
import unittest

from experiments import read_run


class ResearchProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = read_run(9)
        cls.outer_folds = read_run(2)['outputs']['household_folds']

    def test_inner_search_excludes_outer_households(self):
        for fold in self.loop['outputs']['outer_details']:
            held_out = {hid for hid, number in self.outer_folds.items() if number == fold['outer_fold']}
            for trial in fold['trials']:
                saved = read_run(trial['run_id'])
                self.assertFalse(held_out & set(saved['inputs']['inner_household_folds']))
                self.assertEqual(saved['inputs']['parent_run'], self.loop['id'])

    def test_every_trial_has_a_prior_expectation_and_saved_models(self):
        for trial_id in self.loop['outputs']['trial_ids']:
            saved = read_run(trial_id)
            self.assertTrue(saved['expectation'])
            self.assertLess(saved['created_at'], saved['finished_at'])
            self.assertEqual(len(saved['outputs']['inner_models']), 3)
            self.assertEqual(saved['status'], 'completed')

    def test_budget_and_rejection_decisions_are_auditable(self):
        for fold in self.loop['outputs']['outer_details']:
            self.assertLessEqual(len(fold['trials']), 3)
            self.assertEqual(fold['trials'][0]['outcome'], 'baseline')
            for trial in fold['trials'][1:]:
                saved = read_run(trial['run_id'])['outputs']
                if saved['outcome'] == 'kept':
                    self.assertGreaterEqual(saved['gain_over_incumbent'], 0.002)
                    self.assertTrue(saved['checks_passed'])


if __name__ == '__main__':
    unittest.main()
