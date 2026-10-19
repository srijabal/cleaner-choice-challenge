"""A finite, nested household-CV model tournament; never reads check outcomes.

Run with: .venv/bin/python model_tournament.py
The existing prototype is frozen. This comparison is exploratory because the
build households and folds have already informed earlier research.
"""
import hashlib
import json
import math
import random
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from choice_model import CHOICES, examples, features, fit, metrics, predict, score_rows
from experiments import begin, finish, paired_interval, read_run, register, versions
from frequency_baseline import fit_frequency
from structured_model import fit as fit_utility

ROOT = Path(__file__).resolve().parent
TIE_TOLERANCE = 0.002


def configurations():
    candidates = [{'id': 'frequency', 'family': 'frequency', 'history': False, 'complexity': 0}]
    for history in [False, True]:
        suffix = 'history' if history else 'current'
        for c in [0.001, 0.01, 0.1]:
            candidates.append({'id': f'logistic_{suffix}_{c}', 'family': 'logistic',
                               'history': history, 'C': c, 'complexity': 2 + int(history)})
        candidates.append({'id': f'utility_{suffix}', 'family': 'utility',
                           'history': history, 'penalty': 0.01, 'complexity': 1 + int(history)})
        for leaf in [20, 50]:
            candidates.append({'id': f'forest_{suffix}_{leaf}', 'family': 'forest',
                               'history': history, 'min_samples_leaf': leaf, 'complexity': 5 + int(history)})
        for leaves in [3, 7]:
            candidates.append({'id': f'boosting_{suffix}_{leaves}', 'family': 'boosting',
                               'history': history, 'max_leaf_nodes': leaves, 'complexity': 7 + int(history)})
    return candidates


def choose(scored):
    """Favor simplicity within a prespecified tiny inner-loss tolerance."""
    best = min(item['log_loss'] for item in scored)
    eligible = [item for item in scored if item['log_loss'] <= best + TIE_TOLERANCE]
    return min(eligible, key=lambda item: (item['config']['complexity'], item['log_loss'], item['config']['id']))['config']


def fit_candidate(config, training):
    family = config['family']
    if family == 'frequency':
        return fit_frequency(training)
    if family == 'logistic':
        return fit(training, config['history'], C=config['C'])
    if family == 'utility':
        return fit_utility(training, config['history'], penalty=config['penalty'])
    x = [features(row['household'], row['visit'], row['history'], config['history']) for row in training]
    y = [row['visit']['choice'] for row in training]
    if family == 'forest':
        model = RandomForestClassifier(n_estimators=80, max_depth=5,
                                      min_samples_leaf=config['min_samples_leaf'],
                                      random_state=42, n_jobs=1)
    elif family == 'boosting':
        model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.05,
                                              max_leaf_nodes=config['max_leaf_nodes'],
                                              min_samples_leaf=30, l2_regularization=1,
                                              early_stopping=False, random_state=42)
    else:
        raise ValueError(f'Unknown family: {family}')
    model.fit(x, y)
    return model


def records_from_probabilities(rows, probabilities):
    records = []
    for row, values in zip(rows, probabilities):
        v, h = row['visit'], row['household']
        probs = {choice: float(values[i]) for i, choice in enumerate(CHOICES)}
        if any(not math.isfinite(p) or p < 0 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-8:
            raise ValueError('Invalid predicted probabilities')
        choice = v['choice']
        buy_probability = 1 - probs['none']
        records.append({'household_id': v['household_id'], 'visit_id': v['visit_id'],
                        'visit_number': int(v['visit_number']), 'choice': choice,
                        'budget': h['budget'], 'channel': v['channel'], 'probabilities': probs,
                        'loss': -math.log(max(probs[choice], 1e-15)),
                        'buy_loss': -math.log(max(buy_probability if choice != 'none' else probs['none'], 1e-15)),
                        'brand_loss': -math.log(max(probs[choice] / max(buy_probability, 1e-15), 1e-15)) if choice != 'none' else None,
                        'brier': sum((probs[p] - float(p == choice))**2 for p in CHOICES),
                        'correct': max(probs, key=probs.get) == choice})
    return records


def score_candidate(config, model, rows):
    if config['family'] in {'frequency', 'utility', 'logistic'}:
        return score_rows(model, rows)
    x = [features(row['household'], row['visit'], row['history'], config['history']) for row in rows]
    raw = model.predict_proba(x)
    # Explicit alignment prevents missing/reordered training classes changing labels.
    aligned = np.zeros((len(rows), len(CHOICES)))
    for column, label in enumerate(model.classes_):
        aligned[:, CHOICES.index(label)] = raw[:, column]
    return records_from_probabilities(rows, aligned)


def inner_folds(training, outer):
    ids = sorted({r['visit']['household_id'] for r in training})
    random.Random(2718 + outer).shuffle(ids)
    return {hid: i % 3 for i, hid in enumerate(ids)}


def run():
    started = time.monotonic()
    candidates = configurations()
    hashes = {}
    for name in ['model_tournament.py', 'choice_model.py', 'structured_model.py',
                 'frequency_baseline.py', 'experiments.py']:
        content = (ROOT / name).read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        (ROOT / 'artifacts/sources' / (sha + '.py')).write_bytes(content)
        hashes[name] = sha
    run_id = register('model_tournament',
        'After preparing the submission I wanted to check whether a wider model search could beat the small utility model. Trees can learn interactions that the additive model misses, but with 200 households I expect a real risk of overfitting. Compare 17 fixed configurations using nested household validation, retain failures, and prefer simplicity within 0.002 inner log loss. No check outcomes or automatic prototype replacement.',
        {'configurations': candidates, 'outer_folds_from_run': 2, 'inner_folds': 3,
         'inner_seed': 2718, 'selection_metric': 'inner held-out log loss',
         'tie_tolerance': TIE_TOLERANCE, 'seed': 42, 'source_sha256': hashes,
         'data': 'visits.csv only; no household IDs or post-choice text as features',
         'budget': '17 settings x 3 inner folds x 5 outer folds; no adaptive expansion',
         'scope': 'Exploratory reused build folds; best among tried candidates, not a global best model.',
         'prototype_policy': 'Do not replace or retune the frozen submission model.'})
    begin(run_id)
    try:
        rows = examples('visits.csv')
        original = read_run(2)
        assert original['data_version'] == versions()
        folds = original['outputs']['household_folds']
        families = sorted({c['family'] for c in candidates})
        records = {family: [] for family in families + ['nested_search', 'starter']}
        trials, details = [], []
        for outer in range(5):
            training = [r for r in rows if folds[r['visit']['household_id']] != outer]
            held_out = [r for r in rows if folds[r['visit']['household_id']] == outer]
            inner = inner_folds(training, outer)
            assert not set(inner) & {r['visit']['household_id'] for r in held_out}
            scored = []
            for config in candidates:
                trial = register('model_tournament_trial',
                    f"Measure {config['id']} on inner households before seeing this outer fold's results.",
                    {'parent_run': run_id, 'outer_fold': outer, 'config': config,
                     'inner_household_folds': inner, 'source_sha256': hashes})
                begin(trial)
                trials.append(trial)
                try:
                    inner_predictions, fold_metrics = [], []
                    for fold in range(3):
                        train = [r for r in training if inner[r['visit']['household_id']] != fold]
                        valid = [r for r in training if inner[r['visit']['household_id']] == fold]
                        model = fit_candidate(config, train)
                        predicted = score_candidate(config, model, valid)
                        inner_predictions.extend(predicted)
                        fold_metrics.append(metrics(predicted))
                    measured = metrics(inner_predictions)
                    finish(trial, {'metrics': measured, 'inner_fold_metrics': fold_metrics,
                                   'outcome': 'measured', 'reproduce': 'Refit from saved configuration, source hashes, data and inner household assignments.'})
                    scored.append({'config': config, 'log_loss': measured['log_loss'], 'trial_id': trial})
                except Exception:
                    finish(trial, {'error': traceback.format_exc(), 'outcome': 'failed'}, 'failed')
            if not scored:
                raise RuntimeError('All candidates failed')
            selected = choose(scored)
            family_choices = {family: choose([item for item in scored if item['config']['family'] == family])
                              for family in families if any(item['config']['family'] == family for item in scored)}
            # No outer outcomes are inspected until all inner trials finish.
            family_predictions = {}
            for family, config in family_choices.items():
                model = fit_candidate(config, training)
                family_predictions[family] = score_candidate(config, model, held_out)
                records[family].extend(family_predictions[family])
            records['nested_search'].extend(family_predictions[selected['family']])
            records['starter'].extend(score_rows('starter', held_out))
            details.append({'outer_fold': outer, 'selected_config': selected,
                            'family_choices': family_choices, 'inner_scores': scored})
            print(f'Finished outer fold {outer + 1}/5; inner choice: {selected["id"]}', flush=True)
        measured = {name: metrics(values) for name, values in records.items() if len(values) == len(rows)}
        incumbent = read_run(7)['outputs']['predictions']['utility_current']
        differences = {name: paired_interval(values, incumbent) for name, values in records.items() if len(values) == len(rows)}
        output = {'metrics': measured, 'minus_frozen_utility': differences,
                  'best_observed_family': min(families, key=lambda f: measured.get(f, {}).get('log_loss', math.inf)),
                  'selected_counts': dict(Counter(item['selected_config']['id'] for item in details)),
                  'outer_details': details, 'trial_ids': trials, 'predictions': records,
                  'elapsed_seconds': time.monotonic() - started,
                  'prototype_changed': False,
                  'limits': 'Nested folds limit selection bias within this experiment, but build data and outer folds were reused after earlier research. Family rankings and bootstrap intervals are exploratory. Tree models have no imposed own-price monotonicity or additive explanation guarantee. No new check set was used.'}
        finish(run_id, output)
        (ROOT / 'artifacts/tournament_summary.json').write_text(json.dumps(
            {'run_id': run_id, **{key: output[key] for key in ['metrics', 'minus_frozen_utility',
              'best_observed_family', 'selected_counts', 'elapsed_seconds', 'prototype_changed', 'limits']}}, indent=2) + '\n')
        print(json.dumps({'run_id': run_id, 'metrics': measured, 'selected_counts': output['selected_counts'],
                          'seconds': output['elapsed_seconds']}, indent=2), flush=True)
        return run_id
    except Exception:
        finish(run_id, {'error': traceback.format_exc()}, 'failed')
        raise


if __name__ == '__main__':
    run()
