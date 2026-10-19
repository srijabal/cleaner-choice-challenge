"""Nested household CV: tune shrinkage without using outer validation outcomes."""
import hashlib
import json
import random
import traceback
from pathlib import Path
from choice_model import examples, fit, score_rows, metrics
from experiments import register, begin, finish, read_run, paired_interval

C_VALUES = [0.01, 0.1, 1.0]


def run():
    source = Path(__file__).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    run_id = register(
        'regularization_nested_cv',
        'After runs 2 and 3, I suspect flexible learned coefficients fit noise. Stronger regularization (smaller C) may improve build-household log loss. It may still fail to beat the starter, and it may not rescue history. Tune C only within each outer training set using inner household folds. This is exploratory follow-up after seeing check results, not fresh confirmation; do not evaluate the check set again.',
        {'C_values': C_VALUES, 'outer_folds_from_run': 2, 'inner_folds': 3,
         'inner_seed': 123, 'variants': ['current_only', 'with_history'],
         'primary_metric': 'outer-fold log loss',
         'selection_rule': 'Choose lowest inner-CV log loss per variant per outer fold; smaller C wins exact ties. Claim no demonstrated improvement if paired household interval includes zero.',
         'data': 'build visits only', 'script_sha256': source_sha})
    begin(run_id)
    try:
        rows = examples('visits.csv')
        original = read_run(2)['outputs']
        folds = original['household_folds']
        records = {'current_only': [], 'with_history': []}
        fold_details = []
        for outer in range(5):
            training = [r for r in rows if folds[r['visit']['household_id']] != outer]
            validation = [r for r in rows if folds[r['visit']['household_id']] == outer]
            ids = sorted({r['visit']['household_id'] for r in training})
            random.Random(123 + outer).shuffle(ids)
            inner = {hid: i % 3 for i, hid in enumerate(ids)}
            detail = {'outer_fold': outer, 'inner_household_folds': inner, 'variants': {}}
            for name, history in [('current_only', False), ('with_history', True)]:
                losses = {}
                for C in C_VALUES:
                    inner_predictions = []
                    for fold in range(3):
                        train = [r for r in training if inner[r['visit']['household_id']] != fold]
                        valid = [r for r in training if inner[r['visit']['household_id']] == fold]
                        inner_predictions.extend(score_rows(fit(train, history, C=C), valid))
                    losses[C] = metrics(inner_predictions)['log_loss']
                chosen = min(C_VALUES, key=lambda C: (losses[C], C))
                model = fit(training, history, C=chosen)
                predictions = score_rows(model, validation)
                records[name].extend(predictions)
                detail['variants'][name] = {'inner_losses': losses, 'selected_C': chosen,
                                            'outer_metrics': metrics(predictions), 'model': model}
            fold_details.append(detail)
        output = {'metrics': {k: metrics(v) for k, v in records.items()},
                  'fold_details': fold_details, 'predictions': records,
                  'comparisons': {}}
        for name in records:
            output['comparisons'][name + '_minus_original_C1'] = paired_interval(records[name], original['predictions'][name])
            output['comparisons'][name + '_minus_starter'] = paired_interval(records[name], original['predictions']['starter'])
        output['comparisons']['tuned_history_minus_tuned_current'] = paired_interval(records['with_history'], records['current_only'])
        output['limitations'] = 'Nested household CV controls the C search within this experiment. This follow-up hypothesis was chosen after earlier results, including the check set. It is not independent confirmation. Intervals hold fitted predictions fixed.'
        finish(run_id, output)
        Path('artifacts/sources', source_sha + '.py').write_bytes(source)
        print(json.dumps({'run_id': run_id, 'log_loss': {k: v['log_loss'] for k, v in output['metrics'].items()},
                          'comparisons': output['comparisons'],
                          'selected_C': [{k: v['selected_C'] for k, v in d['variants'].items()} for d in fold_details]}, indent=2))
    except Exception:
        finish(run_id, {'error': traceback.format_exc()}, 'failed')
        raise


if __name__ == '__main__':
    run()
