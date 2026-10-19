"""Reproducible experiment ledger. Expectations are committed to SQLite before execution."""
import argparse
import datetime as dt
import hashlib
import json
import random
import sqlite3
import traceback
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from choice_model import examples, fit, score_rows, metrics

ROOT=Path(__file__).resolve().parent
DB=ROOT/'artifacts/runs.sqlite'


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def dumps(x): return json.dumps(x,allow_nan=False)
def versions():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*.csv'))}
def source_version():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['choice_model.py','experiments.py','baseline.py']}
@contextmanager
def connection():
    DB.parent.mkdir(exist_ok=True)
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute('CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, kind TEXT NOT NULL, expectation TEXT NOT NULL, inputs TEXT NOT NULL, outputs TEXT, model_version TEXT NOT NULL, data_version TEXT NOT NULL)')
    try:
        with c:
            yield c
    finally:
        c.close()

def register(kind, expectation, inputs):
    with connection() as c:
        cur=c.execute('INSERT INTO runs(created_at,status,kind,expectation,inputs,model_version,data_version) VALUES(?,?,?,?,?,?,?)',
           (now(),'planned',kind,expectation,dumps(inputs),dumps(source_version()),dumps(versions())))
        return cur.lastrowid

def read_run(run_id):
    with connection() as c:
        r=c.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
    if not r: raise ValueError('Unknown run ID')
    out=dict(r)
    for key in ['inputs','outputs','data_version']:
        if out[key]: out[key]=json.loads(out[key])
    return out

def begin(run_id):
    run=read_run(run_id)
    if run['status']!='planned': raise ValueError('Runs are immutable once started. Register a new run instead.')
    if run['data_version'] != versions(): raise ValueError('Data changed after registration; register a new run.')
    with connection() as c:
        c.execute('UPDATE runs SET status=?,model_version=? WHERE id=?',('running',dumps(source_version()),run_id))
    return run

def finish(run_id, output, status='completed'):
    with connection() as c:
        c.execute('UPDATE runs SET status=?,finished_at=?,outputs=? WHERE id=?',(status,now(),dumps(output),run_id))

def paired_interval(first, second, key='loss', seed=42):
    # Whole-household resampling preserves dependence among the six visits.
    groups=defaultdict(list)
    assert [r['visit_id'] for r in first]==[r['visit_id'] for r in second]
    for a,b in zip(first,second): groups[a['household_id']].append(a[key]-b[key])
    means=np.array([sum(v)/len(v) for v in groups.values()])
    rng=np.random.default_rng(seed)
    boot=means[rng.integers(0,len(means),size=(2000,len(means)))].mean(axis=1)
    return {'mean':float(means.mean()),'ci95':[float(v) for v in np.quantile(boot,[.025,.975])],
            'households':len(means),'interpretation':'negative favors first model; descriptive bootstrap with fitted predictions held fixed'}


def comparison(run):
    rows=examples('visits.csv'); ids=sorted({r['visit']['household_id'] for r in rows})
    random.Random(run['inputs']['seed']).shuffle(ids)
    folds={hid:i%5 for i,hid in enumerate(ids)}
    all_records={m:[] for m in ['starter','current_only','with_history']}; fold_results=[]; saved_models=[]
    for fold in range(5):
        train=[r for r in rows if folds[r['visit']['household_id']]!=fold]
        test=[r for r in rows if folds[r['visit']['household_id']]==fold]
        models={'starter':'starter','current_only':fit(train,False),'with_history':fit(train,True)}
        fold_results.append({'fold':fold,'train_households':160,'test_households':40,'metrics':{}})
        for name,model in models.items():
            rec=score_rows(model,test); all_records[name]+=rec
            fold_results[-1]['metrics'][name]=metrics(rec)
        saved_models.append({'fold':fold,'models':models})
    delta=paired_interval(all_records['with_history'],all_records['current_only'])
    selected='with_history' if delta['ci95'][1]<0 else 'current_only'
    return {'metrics':{k:metrics(v) for k,v in all_records.items()},'folds':fold_results,
            'history_minus_current_log_loss':delta,
            'history_minus_current_buy_log_loss':paired_interval(all_records['with_history'],all_records['current_only'],'buy_loss'),
            'selected_model':selected, 'household_folds':folds,'fold_models':saved_models,'predictions':all_records,
            'caveat':'Cross-validation folds share training observations; this bootstrap is descriptive, not a causal or definitive significance test.'}


def check(run):
    selected=run['inputs']['selected_model']
    training=examples('visits.csv'); testing=examples('check_visits.csv')
    assert not ({r['visit']['household_id'] for r in training}&{r['visit']['household_id'] for r in testing})
    models={'starter':'starter','current_only':fit(training,False),'with_history':fit(training,True)}
    rec={k:score_rows(v,testing) for k,v in models.items()}
    output={'selected_model':selected,'models':models,'metrics':{k:metrics(v) for k,v in rec.items()},
            'history_minus_current_log_loss':paired_interval(rec['with_history'],rec['current_only']),
            'history_minus_current_buy_log_loss':paired_interval(rec['with_history'],rec['current_only'],'buy_loss'),
            'predictions':rec,'slices':{}}
    for field in ['budget','channel','visit_number']:
        output['slices'][field]={str(value):metrics([r for r in rec[selected] if r[field]==value]) for value in sorted({r[field] for r in rec[selected]})}
    return output


def execute(run_id):
    run=begin(run_id)
    try:
        output={'history_comparison':comparison,'frozen_check':check}[run['kind']](run)
        finish(run_id,output)
        print(dumps({'run_id':run_id,**{k:v for k,v in output.items() if k in ['metrics','selected_model','history_minus_current_log_loss','history_minus_current_buy_log_loss']}}))
    except Exception:
        finish(run_id,{'error':traceback.format_exc()},'failed'); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=int);p.add_argument('--show',type=int)
    a=p.parse_args()
    if a.run: execute(a.run)
    elif a.show: print(json.dumps(read_run(a.show),indent=2))
    else:
        with connection() as c:
            for r in c.execute('SELECT id,kind,status,created_at FROM runs'): print(dict(r))
