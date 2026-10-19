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
