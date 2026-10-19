"""Evaluate a restrained model designed for interpretable offer scenarios."""
import hashlib
import json
import traceback
from pathlib import Path

from choice_model import examples, metrics, score_rows
from structured_model import fit
from experiments import register, begin, finish, read_run, paired_interval, versions


def run():
    sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
               for name in ['structured_model.py', 'structured_experiment.py', 'choice_model.py']}
    run_id = register(
        'shared_utility_comparison',
        'The generic multinomial model learns many separate effects for each choice. A smaller shared-utility model may generalize as well or better while giving coherent price responses and simpler explanations. Compare 11 current-only parameters and 13 with history against saved tuned generic and frequency baselines on the same build household folds. History may still offer no reliable gain. Fixed penalty 0.01, no tuning. This is exploratory development after seeing the original check results; do not access check outcomes.',
        {'folds_from_run': 2, 'comparators_from_runs': [5, 6],
         'penalty': 0.01, 'data': 'visits.csv only', 'source_sha256': sources,
         'selection_rule': 'Select history only if its paired loss difference interval versus utility current-only is entirely below zero. Use the selected utility model for the exploratory demo only if its mean loss is no more than 0.01 worse than tuned generic current-only and its price/explanation checks pass. Otherwise retain tuned generic current-only and report structural-model failure. The 0.01 tolerance is a pragmatic development choice, not a significance threshold.',
         'assumptions': ['nonpositive own-price effects', 'shared price sensitivity across products',
                         'tight budgets may have an extra price penalty', 'additive utility']})
    begin(run_id)
    for name, sha in sources.items():
        Path('artifacts/sources', sha + '.py').write_bytes(Path(name).read_bytes())
    try:
        original = read_run(2)
        assert original['data_version'] == versions()
        folds = original['outputs']['household_folds']
        rows = examples('visits.csv')
        records = {'utility_current': [], 'utility_history': []}
        models = []
        for fold in range(5):
            train = [r for r in rows if folds[r['visit']['household_id']] != fold]
            valid = [r for r in rows if folds[r['visit']['household_id']] == fold]
            assert not ({r['visit']['household_id'] for r in train} & {r['visit']['household_id'] for r in valid})
            fitted = {}
            for name, history in [('utility_current', False), ('utility_history', True)]:
                fitted[name] = fit(train, history, penalty=0.01)
                records[name].extend(score_rows(fitted[name], valid))
            models.append({'fold': fold, 'models': fitted})
        history_delta = paired_interval(records['utility_history'], records['utility_current'])
        selected = 'utility_history' if history_delta['ci95'][1] < 0 else 'utility_current'
        generic = read_run(5)['outputs']['predictions']['current_only']
        difference = paired_interval(records[selected], generic)
        eligible = difference['mean'] <= 0.01
        output = {'metrics': {name: metrics(rec) for name, rec in records.items()},
                  'fold_models': models, 'predictions': records,
                  'history_minus_current': history_delta,
                  'selected_utility_minus_tuned_generic': difference,
                  'selected_utility_minus_frequency': paired_interval(records[selected], read_run(6)['outputs']['predictions']['frequency']),
                  'selected_utility_variant': selected, 'eligible_pending_software_checks': eligible,
                  'final_model': fit(rows, selected == 'utility_history', penalty=0.01),
                  'caveat': 'Reused build folds and adaptive model development. Bootstrap intervals are descriptive; these results are not a new untouched test. Monotonic prices are imposed, not discovered.'}
        finish(run_id, output)
        print(json.dumps({'run_id': run_id, 'metrics': output['metrics'],
                          'history_minus_current': history_delta,
                          'selected_utility_minus_tuned_generic': difference,
                          'selected_utility_variant': selected, 'eligible_pending_software_checks': eligible,
                          'weights': dict(zip(output['final_model']['feature_names'], output['final_model']['weights']))}, indent=2))
    except Exception:
        finish(run_id, {'error': traceback.format_exc()}, 'failed')
        raise


if __name__ == '__main__':
    run()
