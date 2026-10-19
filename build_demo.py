"""Export saved evidence and fitted weights. Does not rerun experiments or fit models."""
import json
import sqlite3
from pathlib import Path
from choice_model import examples, predict
from experiments import read_run, DB

root=Path(__file__).resolve().parent
out=root/'demo';out.mkdir(exist_ok=True)
check=read_run(3);build=read_run(2)
with sqlite3.connect(DB) as c: ids=[r[0] for r in c.execute('SELECT id FROM runs ORDER BY id')]
runs=[read_run(i) for i in ids]
(out/'runs.json').write_text(json.dumps(runs,allow_nan=False))
# SQLite backup API makes a consistent downloadable copy.
with sqlite3.connect(DB) as source,sqlite3.connect(out/'runs.sqlite') as target: source.backup(target)
rows=examples('visits.csv')
model=check['outputs']['models']['current_only']
payload={'model':model,'history_model':check['outputs']['models']['with_history'],
         'examples':[{'household':r['household'],'visit':r['visit'],'history':r['history']} for r in rows[:60]],
         'metrics':check['outputs']['metrics'], 'source_run':3,'data_version':check['data_version'],
         'parity_cases':[{'input':r,'expected':predict(model,r['household'],r['visit'],r['history'])} for r in rows[::29]]}
(out/'model.json').write_text(json.dumps(payload,allow_nan=False))
print('Exported model, example households, runs.json and runs.sqlite')
