"""Post-submission sensitivity check using five new household partitions.

This reuses the 200 build households. Seeds and models are fixed before fitting;
no model, penalty or offer is selected from these results.
"""
import hashlib
import importlib.metadata
import json
import random
import traceback
from pathlib import Path

from choice_model import examples, fit as fit_logistic, metrics, score_rows
from experiments import begin, finish, paired_interval, read_run, register, versions
from frequency_baseline import fit_frequency
from structured_model import fit

ROOT = Path(__file__).resolve().parent
SEEDS = [101, 202, 303, 404, 505]
MODELS = ['utility_current', 'utility_history', 'logistic_current', 'frequency']


def household_folds(rows, seed):
    ids = sorted({row['visit']['household_id'] for row in rows})
    random.Random(seed).shuffle(ids)
    return {hid: i % 5 for i, hid in enumerate(ids)}


def average_losses(repeats, name):
    """Average losses per visit, not predictions from a new ensemble."""
    ordered = [sorted(item['predictions'][name], key=lambda r: r['visit_id']) for item in repeats]
    result = []
    for group in zip(*ordered):
        assert len({r['visit_id'] for r in group}) == 1
        result.append({'visit_id': group[0]['visit_id'], 'household_id': group[0]['household_id'],
                       **{key: sum(r[key] for r in group) / len(group) for key in ['loss', 'buy_loss']}})
    return result


def run():
    hashes = {}
    for name in ['split_robustness.py', 'structured_model.py', 'choice_model.py',
                 'frequency_baseline.py', 'experiments.py']:
        content = (ROOT / name).read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        (ROOT / 'artifacts/sources' / (sha + '.py')).write_bytes(content)
        hashes[name] = sha
    run_id = register('split_robustness',
        'Follow-up after submission: did the earlier result depend on the particular household split? I expect the small utility model to remain better than average choice frequencies for identifying the chosen cleaner, but not necessarily for buy versus skip. The benefit of history may change across splits. Test five fixed new seeds without tuning or replacing the submitted model. These are the same build households, not new independent evidence.',
        {'seeds': SEEDS, 'folds': 5, 'models': MODELS, 'utility_penalty': 0.01,
         'logistic_C': 0.01, 'data': 'visits.csv only', 'source_sha256': hashes,
         'environment': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn']},
         'timing': 'post-submission follow-up',
         'policy': 'Report every seed, including unfavorable results. No retuning, model selection, or changes to submitted files.',
         'summary': 'Mean and range across seeds describe split sensitivity. Seeds are not independent samples. Bootstrap averages loss per visit across seeds, then resamples whole households with predictions fixed.'})
    begin(run_id)
    try:
        rows = examples('visits.csv')
        assert read_run(2)['data_version'] == versions()
        repeats, trial_ids = [], []
        for seed in SEEDS:
            folds = household_folds(rows, seed)
            trial = register('split_robustness_trial',
                f'Check the fixed models on household partition seed {seed}; do not change settings after seeing outcomes.',
                {'parent_run': run_id, 'seed': seed, 'household_folds': folds,
                 'models': MODELS, 'utility_penalty': 0.01, 'logistic_C': 0.01,
                 'source_sha256': hashes})
            trial_ids.append(trial)
            begin(trial)
            try:
                predictions = {name: [] for name in MODELS}
                fold_models = []
                for fold in range(5):
                    training = [r for r in rows if folds[r['visit']['household_id']] != fold]
                    held_out = [r for r in rows if folds[r['visit']['household_id']] == fold]
                    assert not ({r['visit']['household_id'] for r in training} &
                                {r['visit']['household_id'] for r in held_out})
                    models = {'utility_current': fit(training, False, penalty=0.01),
                              'utility_history': fit(training, True, penalty=0.01),
                              'logistic_current': fit_logistic(training, False, C=0.01),
                              'frequency': fit_frequency(training)}
                    for name, model in models.items():
                        predictions[name].extend(score_rows(model, held_out))
                    fold_models.append({'fold': fold, 'models': models})
                result = {'seed': seed, 'metrics': {k: metrics(v) for k, v in predictions.items()},
                          'predictions': predictions, 'fold_models': fold_models}
                finish(trial, result)
                repeats.append(result)
                print(f'Completed seed {seed}', flush=True)
            except Exception:
                finish(trial, {'error': traceback.format_exc()}, 'failed')
                raise
        averaged = {name: average_losses(repeats, name) for name in MODELS}
        overview = {}
        for name in MODELS:
            overview[name] = {}
            for key in ['log_loss', 'buy_log_loss', 'accuracy']:
                values = [r['metrics'][name][key] for r in repeats]
                overview[name][key] = {'mean': sum(values)/len(values), 'range': [min(values), max(values)]}
        comparisons = {}
        for name in ['utility_history', 'logistic_current', 'frequency']:
            comparisons[name + '_minus_utility_current'] = {
                key: paired_interval(averaged[name], averaged['utility_current'], key)
                for key in ['loss', 'buy_loss']}
        output = {'overview': overview, 'comparisons': comparisons, 'trial_ids': trial_ids,
                  'per_seed': [{'seed': r['seed'], 'metrics': r['metrics']} for r in repeats],
                  'submitted_model_changed': False,
                  'limits': 'Same 200 build households, repeatedly partitioned. Ranges are not confidence intervals. Household bootstrap intervals average repeat losses with predictions fixed; they omit adaptive research and full training uncertainty. No check outcomes or fresh households were used.'}
        finish(run_id, output)
        summary = {'run_id': run_id, **output}
        (ROOT / 'artifacts/split_robustness_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'run_id': run_id, 'overview': overview, 'comparisons': comparisons}, indent=2))
        return run_id
    except Exception:
        finish(run_id, {'error': traceback.format_exc()}, 'failed')
        raise


if __name__ == '__main__':
    run()
