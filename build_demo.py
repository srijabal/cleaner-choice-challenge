"""Export frozen evidence and the chosen prototype without rerunning experiments."""
from contextlib import closing
import json
import sqlite3
from pathlib import Path

from choice_model import examples, predict
from experiments import read_run, DB

root = Path(__file__).resolve().parent
out = root / 'demo'
out.mkdir(exist_ok=True)
with closing(sqlite3.connect(DB)) as connection:
    ids = [r[0] for r in connection.execute('SELECT id FROM runs ORDER BY id')]
runs = [read_run(i) for i in ids]
(out / 'runs.json').write_text(json.dumps(runs, allow_nan=False))
with closing(sqlite3.connect(DB)) as source, closing(sqlite3.connect(out / 'runs.sqlite')) as target:
    source.backup(target)

decision = read_run(8)
model = decision['outputs']['model']
rows = examples('visits.csv')
payload = {
    'model': model, 'source_run': 8, 'data_version': decision['data_version'],
    'population': [r['household'] for r in rows if int(r['visit']['visit_number']) == 1],
    'parity_cases': [{'input': r, 'expected': predict(model, r['household'], r['visit'], r['history'])}
                     for r in rows[::29]],

}
(out / 'model.json').write_text(json.dumps(payload, allow_nan=False))
print(f'Exported final prototype from run 8; {len(runs)} saved runs retained.')

# Keep the initial page small; full evidence remains available by run ID and SQLite.
run_folder = out / 'runs'
run_folder.mkdir(exist_ok=True)
index = []
for run in runs:
    (run_folder / f"{run['id']}.json").write_text(json.dumps(run, allow_nan=False))
    output = run.get('outputs') or {}
    index.append({**{key: run[key] for key in ['id', 'kind', 'status', 'expectation', 'created_at', 'finished_at']},
                  'summary': {key: output[key] for key in ['decision', 'outcome', 'error'] if key in output}})
(out / 'runs.json').write_text(json.dumps(index, allow_nan=False))

def score(label, run_id, key, selected=False):
    return {'label': label, 'run_id': run_id, 'loss': read_run(run_id)['outputs']['metrics'][key]['log_loss'], 'selected': selected}

graphs = {'journey': {'rows': [
    score('Average choice rates', 6, 'frequency'), score('Starter rules', 2, 'starter'),
    score('Initial logistic', 2, 'current_only'), score('Regularized logistic', 5, 'current_only'),
    score('Final utility model', 7, 'utility_current', True)],
    'caption': 'Same 200 build households; exploratory comparisons on reused household folds.'}}
tournaments = [r for r in runs if r['kind'] == 'model_tournament' and r['status'] == 'completed']
if tournaments:
    latest = tournaments[-1]
    names = {'frequency': 'Average choice rates', 'starter': 'Starter rules', 'logistic': 'Logistic regression',
             'utility': 'Shared utility', 'forest': 'Random forest', 'boosting': 'Boosted trees', 'nested_search': 'Nested model search'}
    graphs['tournament'] = {'rows': [{'label': names[key], 'run_id': latest['id'], 'loss': value['log_loss'], 'selected': key == 'utility'}
                                      for key, value in latest['outputs']['metrics'].items()],
                            'caption': f"Run #{latest['id']}: 17 configurations, inner tuning and outer household evaluation. Reused build data."}
(out / 'graphs.json').write_text(json.dumps(graphs, allow_nan=False))
