"""A bounded, rule-based research loop inspired by the autoresearch pattern.

No agent or LLM writes code here. This controller proposes a short sequence of
predefined regularization changes, records expectations before execution, and
keeps or rejects each change using inner household validation. Outer households
are evaluated only after that fold's search stops. The original check file is
never loaded. Reused build data still make this an exploratory follow-up.
"""
import argparse
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

from choice_model import examples, metrics, predict, score_rows
from experiments import begin, finish, paired_interval, read_run, register
from structured_model import fit

MAX_SETTINGS = 3
PATIENCE = 2
MIN_GAIN = 0.002


def source_hashes():
    names = ['research_loop.py', 'structured_model.py', 'choice_model.py', 'experiments.py']
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    for name, sha in hashes.items():
        Path('artifacts/sources', sha + '.py').write_bytes(Path(name).read_bytes())
    return hashes


def propose(accepted_stronger):
    if accepted_stronger:
        return 0.3, 'Stronger shrinkage helped; a further small increase may help generalization.'
    return 0.001, 'Stronger shrinkage did not help; weaker shrinkage may recover useful offer effects.'


def checks_pass(model, rows):
    for row in rows[::7]:
        result = predict(model, row['household'], row['visit'], [])
        if abs(sum(result['probabilities'].values()) - 1) > 1e-10:
            return False
        if abs(sum(t['contribution'] for t in result['contributions']) - result['log_odds_margin']) > 1e-10:
            return False
        changed = dict(row['visit'], entrant_price=float(row['visit']['entrant_price']) + 1)
        if predict(model, row['household'], changed, [])['probabilities']['entrant'] > result['probabilities']['entrant'] + 1e-12:
            return False
    return True


def run(expectation=None):
    hashes = source_hashes()
    run_id = register(
        'bounded_research_loop',
        expectation or 'A short adaptive search may improve the utility model, but the gain may not survive outer household validation. I expect the loop may correctly stop without replacing the incumbent. No check outcomes, code mutation, API, or GPU.',
        {'max_settings_per_outer_fold': MAX_SETTINGS, 'patience': PATIENCE,
         'baseline_penalty': 0.01, 'first_change': 0.1,
         'next_change': '0.3 if stronger shrinkage was accepted, otherwise 0.001',
         'keep_rule': 'Inner-CV log loss must improve by at least 0.002, buy/no-buy loss must not worsen by more than 0.005, and model checks must pass.',
         'replacement_rule': 'Do not replace the existing prototype unless the descriptive 95% paired outer-household interval for loss difference is entirely below zero. This criterion does not remove adaptive-research uncertainty.',
         'outer_folds_from_run': 2, 'inner_folds': 3, 'inner_seed': 314,
         'history': False, 'data': 'visits.csv only', 'source_sha256': hashes,
         'inspiration': 'https://github.com/karpathy/autoresearch',
         'scope': 'Predefined CPU experiments; not an LLM researcher or a guarantee of an optimal model.'})
    begin(run_id)
    started = time.monotonic()
    try:
        rows = examples('visits.csv')
        folds = read_run(2)['outputs']['household_folds']
        outer_predictions, outer_details, trial_ids = [], [], []
        for outer in range(5):
            training = [r for r in rows if folds[r['visit']['household_id']] != outer]
            held_out = [r for r in rows if folds[r['visit']['household_id']] == outer]
            ids = sorted({r['visit']['household_id'] for r in training})
            random.Random(314 + outer).shuffle(ids)
            inner = {hid: i % 3 for i, hid in enumerate(ids)}
            assert not set(inner) & {r['visit']['household_id'] for r in held_out}
            best_metrics, best_penalty, stale = None, 0.01, 0
            accepted_stronger = False
            decisions = []
            for setting in range(MAX_SETTINGS):
                if setting == 0:
                    penalty, hypothesis = 0.01, 'Establish the existing utility model as the fixed inner-validation reference.'
                elif setting == 1:
                    penalty, hypothesis = 0.1, 'A stronger penalty may reduce noise in the learned effects.'
                else:
                    penalty, hypothesis = propose(accepted_stronger)
                trial_id = register('research_loop_trial', hypothesis,
                                    {'parent_run': run_id, 'outer_fold': outer, 'setting': setting,
                                     'penalty': penalty, 'inner_household_folds': inner,
                                     'source_sha256': hashes, 'data': 'outer training households only'})
                trial_ids.append(trial_id)
                begin(trial_id)
                try:
                    predictions, fitted_models, valid_checks = [], [], True
                    for fold in range(3):
                        train = [r for r in training if inner[r['visit']['household_id']] != fold]
                        valid = [r for r in training if inner[r['visit']['household_id']] == fold]
                        model = fit(train, False, penalty=penalty)
                        predictions.extend(score_rows(model, valid))
                        fitted_models.append(model)
                        valid_checks = valid_checks and checks_pass(model, valid)
                    measured = metrics(predictions)
                    gain = None if best_metrics is None else best_metrics['log_loss'] - measured['log_loss']
                    keep = valid_checks and (best_metrics is None or
                           (gain >= MIN_GAIN and measured['buy_log_loss'] <= best_metrics['buy_log_loss'] + 0.005))
                    outcome = 'baseline' if setting == 0 else 'kept' if keep else 'rejected'
                    finish(trial_id, {'metrics': measured, 'outcome': outcome,
                                      'gain_over_incumbent': gain, 'checks_passed': valid_checks,
                                      'inner_models': fitted_models, 'predictions': predictions})
                    decisions.append({'run_id': trial_id, 'penalty': penalty, 'outcome': outcome,
                                      'log_loss': measured['log_loss'], 'gain': gain})
                    if keep:
                        best_metrics, best_penalty, stale = measured, penalty, 0
                    else:
                        stale += 1
                    if setting == 1:
                        accepted_stronger = keep
                    if stale >= PATIENCE:
                        break
                except Exception:
                    finish(trial_id, {'error': traceback.format_exc(), 'outcome': 'failed'}, 'failed')
                    raise
            # Outer outcomes become available only after the search is finished.
            selected = fit(training, False, penalty=best_penalty)
            outer_predictions.extend(score_rows(selected, held_out))
            outer_details.append({'outer_fold': outer, 'selected_penalty': best_penalty,
                                  'stop_reason': 'two_changes_without_enough_gain' if stale >= PATIENCE else 'setting_budget_reached',
                                  'trials': decisions, 'model': selected})
        incumbent = read_run(7)['outputs']['predictions']['utility_current']
        delta = paired_interval(outer_predictions, incumbent)
        replace = delta['ci95'][1] < 0
        output = {
            'metrics': metrics(outer_predictions), 'incumbent_metrics': metrics(incumbent),
            'loop_minus_incumbent': delta, 'would_replace_incumbent': replace,
            'decision': 'eligible_for_review' if replace else 'keep_existing_model',
            'outer_details': outer_details, 'trial_ids': trial_ids,
            'predictions': outer_predictions, 'elapsed_seconds': time.monotonic() - started,
            'note': 'No model export was changed automatically. The build folds were reused after prior results. Outer evaluation limits within-loop selection bias but is not fresh external confirmation.'}
        finish(run_id, output)
        print(json.dumps({'run_id': run_id, 'decision': output['decision'],
                          'log_loss': output['metrics']['log_loss'], 'delta': delta,
                          'trial_ids': trial_ids, 'seconds': output['elapsed_seconds']}, indent=2))
        return run_id
    except Exception:
        finish(run_id, {'error': traceback.format_exc()}, 'failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expectation')
    args = parser.parse_args()
    run(args.expectation)
