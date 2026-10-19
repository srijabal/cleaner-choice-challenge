import copy
import math
import unittest
from choice_model import examples, features, fit, predict


class ChoiceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=examples('visits.csv')
        cls.model=fit(cls.rows,True)

    def test_current_outcomes_and_text_cannot_change_features(self):
        r=self.rows[5]; v=copy.deepcopy(r['visit'])
        v.update(choice='scented',said_why='I love scent',after_use_note='It worked')
        self.assertEqual(features(r['household'],r['visit'],r['history'],True),features(r['household'],v,r['history'],True))

    def test_history_contains_only_earlier_same_household_visits(self):
        for row in self.rows:
            self.assertTrue(all(p['household_id']==row['visit']['household_id'] and int(p['visit_number'])<int(row['visit']['visit_number']) for p in row['history']))

    def test_probability_and_explanation_arithmetic(self):
        for r in self.rows[::13]:
            p=predict(self.model,r['household'],r['visit'],r['history'])
            self.assertAlmostEqual(sum(p['probabilities'].values()),1)
            self.assertAlmostEqual(sum(t['contribution'] for t in p['contributions']),p['log_odds_margin'])
            self.assertAlmostEqual(math.log(p['probabilities'][p['top_choice']]/p['probabilities'][p['runner_up']]),p['log_odds_margin'])

    def test_current_only_model_ignores_history(self):
        r=self.rows[5]
        self.assertEqual(features(r['household'],r['visit'],[],False),features(r['household'],r['visit'],r['history'],False))

    def test_missing_spending_is_distinct_from_zero(self):
        r=self.rows[0];h=dict(r['household']);h['cleaning_spend_usd']=''
        missing=features(h,r['visit'],[],False)
        h['cleaning_spend_usd']='0'
        self.assertNotEqual(missing,features(h,r['visit'],[],False))

    def test_portable_inference_matches_training_library(self):
        from sklearn.linear_model import LogisticRegression
        rows=self.rows[:300]
        x=[features(r['household'],r['visit'],r['history'],True) for r in rows]
        y=[r['visit']['choice'] for r in rows]
        trained=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,random_state=42).fit(x,y)
        portable=fit(rows,True)
        actual=predict(portable,rows[0]['household'],rows[0]['visit'],rows[0]['history'])['probabilities']
        for label,p in zip(trained.classes_,trained.predict_proba([x[0]])[0]): self.assertAlmostEqual(actual[label],p,places=12)

if __name__=='__main__': unittest.main()
