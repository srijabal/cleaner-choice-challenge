"""Test whether input-dependent models beat training-only choice frequencies."""
import hashlib
import json
import math
import traceback
from collections import Counter
from pathlib import Path

from choice_model import CHOICES, CURRENT_FEATURES, examples, metrics, score_rows
from experiments import begin, finish, paired_interval, read_run, register, versions


def fit_frequency(rows):
    """One smoothed probability per choice, shared by all held-out households."""
    counts = Counter(row['visit']['choice'] for row in rows)
    probabilities = {choice: (counts[choice] + 1) / (len(rows) + len(CHOICES))
                     for choice in CHOICES}
    return {
        'type': 'smoothed_choice_frequency',
        'version': 'frequency-v1',
        'history': False,
        'feature_names': CURRENT_FEATURES,
        'classes': CHOICES,
        'coefficients': [[0.0] * len(CURRENT_FEATURES) for _ in CHOICES],
        'intercepts': [math.log(probabilities[choice]) for choice in CHOICES],
        'training_counts': dict(counts),
        'training_visits': len(rows),
        'laplace_alpha': 1,
        'probabilities': probabilities,
    }


def run():
    source = Path(__file__).read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    run_id = register(
        'frequency_baseline',
        'Run 5 improved when effects were shrunk. A training-only choice-frequency baseline might perform almost as well, suggesting little predictive value in the inputs. I expect tuned models to beat frequencies on average, but the advantage may be uncertain. Use the existing household folds, estimate frequencies only on training households, and compare saved held-out predictions. This is exploratory analysis on reused build data, not a fresh test. Do not access check outcomes.',
        {'folds_from_run': 2, 'tuned_predictions_from_run': 5,
         'data': 'visits.csv only', 'laplace_alpha': 1,
         'primary_metric': 'held-out log loss',
         'comparisons': ['starter', 'tuned_current_only', 'tuned_with_history'],
         'decision_rule': 'An interval spanning zero means the advantage is not established. No model reselection or retuning in this run.',
         'script_sha256': sha})
    begin(run_id)
    Path('artifacts/sources', sha + '.py').write_bytes(source)
    try:
        original = read_run(2)
        tuned = read_run(5)
        assert original['data_version'] == tuned['data_version'] == versions()
        rows = examples('visits.csv')
        folds = original['outputs']['household_folds']
        records = []
        details = []
        for fold in range(5):
            training = [r for r in rows if folds[r['visit']['household_id']] != fold]
            validation = [r for r in rows if folds[r['visit']['household_id']] == fold]
            train_ids = {r['visit']['household_id'] for r in training}
            validation_ids = {r['visit']['household_id'] for r in validation}
            assert not train_ids & validation_ids
            assert len(train_ids) == 160 and len(validation_ids) == 40
            model = fit_frequency(training)
            predicted = score_rows(model, validation)
            # The frequency model must give the same probabilities regardless of inputs.
            for row in predicted:
                for choice in CHOICES:
                    assert abs(row['probabilities'][choice] - model['probabilities'][choice]) < 1e-12
            records.extend(predicted)
            details.append({'fold': fold, 'model': model, 'metrics': metrics(predicted)})
        comparators = {
            'starter': original['outputs']['predictions']['starter'],
            'tuned_current_only': tuned['outputs']['predictions']['current_only'],
            'tuned_with_history': tuned['outputs']['predictions']['with_history'],
        }
        output = {
            'metrics': {'frequency': metrics(records),
                        **{k: metrics(v) for k, v in comparators.items()}},
            'comparisons': {name + '_minus_frequency': paired_interval(values, records)
                            for name, values in comparators.items()},
            'buy_comparisons': {name + '_minus_frequency': paired_interval(values, records, 'buy_loss')
                                for name, values in comparators.items()},
            'fold_details': details,
            'predictions': {'frequency': records},
            'limitations': 'An exploratory comparison on reused build folds. Household bootstrap intervals hold fitted predictions fixed and do not account for adaptive experiment choice or all training uncertainty.'}
        finish(run_id, output)
        print(json.dumps({'run_id': run_id,
                          'log_loss': {k: v['log_loss'] for k, v in output['metrics'].items()},
                          'comparisons': output['comparisons'],
                          'buy_comparisons': output['buy_comparisons']}, indent=2))
    except Exception:
        finish(run_id, {'error': traceback.format_exc()}, 'failed')
        raise


if __name__ == '__main__':
    run()
