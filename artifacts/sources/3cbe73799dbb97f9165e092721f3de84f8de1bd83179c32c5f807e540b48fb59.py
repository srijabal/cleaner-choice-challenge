"""Connect the selected model to a bounded offer test, with explicit uncertainty."""
import hashlib
import itertools
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from baseline import predict as starter_predict
from choice_model import examples, fit, predict
from experiments import begin, finish, read_run, register
from run_baseline import offers_for


def evaluate(model, household, visit):
    if model == 'starter':
        return starter_predict(household, offers_for(visit), visit, [])['probabilities']['entrant']
    return predict(model, household, visit, [])['probabilities']['entrant']


def run():
    files = ['decision_experiment.py', 'structured_model.py', 'choice_model.py']
    sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
    run_id = register(
        'final_offer_sensitivity',
        'The selected utility model may prefer $3.49/environment/online for purchase share. I expect lower-price demand gains not to imply the best profit: rankings may change with hypothetical costs, model family or training fold. Evaluate all observed price/message/channel levels on 200 build households, fixed competitor offers, and no counterfactual history. Compare fold fits and model families. Do not infer causal effects, real profit, channel superiority, or claim efficacy. No check outcomes will be accessed.',
        {'model_run': 7, 'prices': [3.49, 4.49, 5.49],
         'messages': ['cleaning', 'low_odor', 'environment'],
         'channels': ['retail', 'online'], 'competitor_prices': [3.99, 2.99, 4.29],
         'hypothetical_unit_costs': [0, 1, 2, 3],
         'source_sha256': sources,
         'decision_rule': 'Report a model-favored test candidate only if the selected utility family and at least 4/5 fold fits agree on the purchase-share winner. Otherwise report an unstable ranking. Always withhold a launch/profit optimum because costs, claim substantiation and external validation are absent. Cost scenarios are assumptions, not supplied data.'})
    begin(run_id)
    for name, sha in sources.items():
        Path('artifacts/sources', sha + '.py').write_bytes(Path(name).read_bytes())
    try:
        chosen = read_run(7)['outputs']
        if not chosen['eligible_pending_software_checks']:
            raise ValueError('Utility candidate failed predeclared predictive-quality threshold.')
        selected = chosen['selected_utility_variant']
        model = chosen['final_model']
        rows = examples('visits.csv')
        first = [r for r in rows if int(r['visit']['visit_number']) == 1]
        models = {'selected_utility': model, 'tuned_generic': fit(rows, False, C=0.01), 'starter': 'starter'}
        models.update({f'utility_fold_{f["fold"]}': f['models'][selected] for f in chosen['fold_models']})
        scenarios = []
        for price, message, channel in itertools.product([3.49, 4.49, 5.49], ['cleaning', 'low_odor', 'environment'], ['retail', 'online']):
            per_model = defaultdict(list)
            household_results = []
            for row in first:
                visit = {**row['visit'], 'entrant_price': price, 'entrant_message': message,
                         'channel': channel, 'standard_price': 3.99, 'budget_price': 2.99, 'scented_price': 4.29}
                for name, candidate in models.items():
                    per_model[name].append(evaluate(candidate, row['household'], visit))
                household_results.append({'household_id': row['household']['household_id'],
                                          'budget': row['household']['budget'],
                                          'entrant_probability': per_model['selected_utility'][-1]})
            means = {name: sum(values) / len(values) for name, values in per_model.items()}
            supported = [r for r in rows if float(r['visit']['entrant_price']) == price
                         and r['visit']['entrant_message'] == message and r['visit']['channel'] == channel]
            fold_means = [value for name, value in means.items() if name.startswith('utility_fold_')]
            scenarios.append({'price': price, 'message': message, 'channel': channel,
                              'predicted_share': means,
                              'fold_share_range': [min(fold_means), max(fold_means)],
                              'support_visits': len(supported),
                              'support_households': len({r['visit']['household_id'] for r in supported}),
                              'households': household_results,
                              'assumed_contribution_per_exposure': {
                                  str(cost): (price - cost) * means['selected_utility'] for cost in [0, 1, 2, 3]}})
        def offer(s):
            return {k: s[k] for k in ['price', 'message', 'channel']}
        winners = {name: max(scenarios, key=lambda s: s['predicted_share'][name]) for name in models}
        candidate = winners['selected_utility']
        agreement = sum(offer(winners[f'utility_fold_{i}']) == offer(candidate) for i in range(5))
        profit_winners = {str(cost): offer(max(scenarios, key=lambda s: s['assumed_contribution_per_exposure'][str(cost)]))
                          for cost in [0, 1, 2, 3]}
        model_checks = {'own_price_nonincreasing': True, 'normalized_probabilities': True,
                        'explanation_reconstructs_log_odds': True, 'households_checked': len(first)}
        for row in first:
            p = predict(model, row['household'], row['visit'], [])
            assert abs(sum(p['probabilities'].values()) - 1) < 1e-10
            assert abs(sum(t['contribution'] for t in p['contributions']) - p['log_odds_margin']) < 1e-10
            v = {**row['visit'], 'entrant_price': float(row['visit']['entrant_price']) + 1}
            assert predict(model, row['household'], v, [])['probabilities']['entrant'] <= p['probabilities']['entrant'] + 1e-12
        output = {'model': model, 'comparator_models': models,
                  'selected_variant': selected, 'scenarios': scenarios,
                  'purchase_share_winners': {name: offer(s) for name, s in winners.items()},
                  'candidate': {**offer(candidate), 'predicted_share': candidate['predicted_share']['selected_utility'],
                                'fold_share_range': candidate['fold_share_range'],
                                'support_visits': candidate['support_visits'],
                                'support_households': candidate['support_households']},
                  'fold_winner_agreement': agreement,
                  'candidate_status': 'stable_within_selected_family' if agreement >= 4 else 'unstable',
                  'hypothetical_profit_winners': profit_winners,
                  'model_checks': model_checks,
                  'decision': 'A provisional shopper-test candidate, not an identified launch optimum. Obtain unit economics and substantiated wording before finalizing the test.',
                  'limits': ['Fold ranges measure training-fit sensitivity, not 95% confidence intervals.',
                             'All scenario estimates are observational model predictions, not causal treatment effects.',
                             'Online has no acquisition or fulfillment cost here; its profit superiority is not established.',
                             'Costs 0, 1, 2, 3 are illustrative assumptions, not real unit costs.',
                             'Current survey and offer information are used; history was not selected.',
                             'Late-stage model development reused build data after the initial check was seen.']}
        finish(run_id, output)
        print(json.dumps({'run_id': run_id, **{k: output[k] for k in [
            'candidate', 'fold_winner_agreement', 'purchase_share_winners',
            'hypothetical_profit_winners', 'model_checks']}}, indent=2))
    except Exception:
        finish(run_id, {'error': traceback.format_exc()}, 'failed')
        raise


if __name__ == '__main__':
    run()
