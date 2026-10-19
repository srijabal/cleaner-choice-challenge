import copy
import math
import unittest

from choice_model import examples, predict
from structured_model import feature_matrix, fit


class StructuredModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = examples('visits.csv')
        cls.model = fit(cls.rows)

    def test_own_price_increase_never_increases_probability(self):
        for row in self.rows[::17]:
            for product in ['entrant', 'standard', 'budget', 'scented']:
                lower = dict(row['visit'])
                higher = dict(lower)
                higher[f'{product}_price'] = float(lower[f'{product}_price']) + 1
                a = predict(self.model, row['household'], lower, row['history'])
                b = predict(self.model, row['household'], higher, row['history'])
                self.assertLessEqual(b['probabilities'][product], a['probabilities'][product] + 1e-12)

    def test_probability_and_explanation_arithmetic(self):
        for row in self.rows[::17]:
            result = predict(self.model, row['household'], row['visit'], row['history'])
            p = result['probabilities']
            self.assertAlmostEqual(sum(p.values()), 1)
            self.assertAlmostEqual(sum(t['contribution'] for t in result['contributions']), result['log_odds_margin'])
            self.assertAlmostEqual(math.log(p[result['top_choice']] / p[result['runner_up']]), result['log_odds_margin'])

    def test_no_current_choice_or_post_choice_text_leakage(self):
        row = self.rows[5]
        edited = copy.deepcopy(row['visit'])
        edited.update(choice='entrant', said_why='Cheap', after_use_note='It worked')
        a = feature_matrix(row['household'], row['visit'], row['history'], True)
        b = feature_matrix(row['household'], edited, row['history'], True)
        self.assertTrue((a == b).all())

    def test_selected_variant_has_no_history_dependency(self):
        row = self.rows[5]
        a = predict(self.model, row['household'], row['visit'], [])
        b = predict(self.model, row['household'], row['visit'], row['history'])
        self.assertEqual(a, b)

    def test_bad_prices_rejected(self):
        row = self.rows[0]
        for price in [-1, 0, float('nan'), float('inf')]:
            v = dict(row['visit'], entrant_price=price)
            with self.assertRaises(ValueError):
                predict(self.model, row['household'], v, [])

    def test_unavailable_survey_spending_not_required(self):
        row = self.rows[0]
        household = {'budget': 'tight', 'usual_brand': ''}
        result = predict(self.model, household, row['visit'], [])
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1)


if __name__ == '__main__':
    unittest.main()
