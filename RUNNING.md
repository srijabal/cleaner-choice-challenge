# Run and inspect

## Local demo — one command

```sh
python3 app.py
```

Open http://127.0.0.1:8000. Python 3.10+; no pip install needed for the demo. It serves the pre-exported fitted weights and recomputes predictions in JavaScript when inputs change. An expectation is saved to browser IndexedDB before prediction. Browser runs can be exported as JSON; they are local to that browser and separate from the research SQLite ledger.

The GitHub repository and hosted demo are private. Reviewers need explicit access or an authorized visibility change before submission. No email or submission has been sent.

## Saved experiments

`artifacts/runs.sqlite` is the authoritative experiment database. `demo/runs.sqlite` is a consistent download copy. `demo/runs.json` is a full export, including failed runs, expectations, inputs, timestamps, file hashes, trained coefficients and per-visit predictions. Select a run ID in the demo; large prediction/weight arrays are omitted from the on-screen view but included in downloads.

Inspect without dependencies:

```sh
python3 -c "import sqlite3; c=sqlite3.connect('artifacts/runs.sqlite'); print(c.execute('SELECT id,kind,status FROM runs').fetchall())"
```

Source snapshots matching every recorded model/source hash are in `artifacts/sources/`. The failed run 1 source was reconstructed from the one-line missing-value fix and verified against its already-recorded SHA-256; it is not presented as a contemporaneously saved snapshot.

## Reproduce training

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py
.venv/bin/python -m unittest discover -s tests -v
```

This appends new runs. It does not overwrite runs 1–4 or pretend the already-inspected check set is fresh. The original failed run cannot be resumed. Its traceback remains available by ID. Hyperparameters and random seeds are fixed; do not interpret replication as independent evidence.

`python3 experiments.py --show 2` prints a full saved run when training dependencies are installed. `python3 build_demo.py` refreshes downloadable artifacts while keeping demo weights tied to frozen run 3. `python3 scenario_test.py` registers and repeats the exploratory scenario comparison. It appends another run and leaves originals unchanged.

## Files

- `FIRST.md`: initial concern committed before model changes.
- `choice_model.py`: fixed features, logistic fit, portable prediction and score contributions.
- `experiments.py`: SQLite ledger, household CV, check evaluation, metrics and bootstrap.
- `scenario_test.py`: exploratory price/message/channel sensitivity.
- `RESULTS.md`: decisions, negative results, limits, household examples and proposed real test.
- `FINAL_DRAFT.md`: under-200-word submission note for review/adaptation.
- `demo/`: static interface, inference code and exported evidence.
- `baseline.py`, `run_baseline.py`, `data/`: original starter code/data, unchanged.

AI assistance was used for implementation, checks and drafting the technical report. FIRST.md records the candidate's initial hypothesis. The candidate should review and be able to explain the code, saved runs and conclusions before submitting.
