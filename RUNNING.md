# Running the work

The simplest command is:

```sh
./run.sh
```

It creates a local virtual environment, installs the CPU dependencies and starts http://127.0.0.1:8000. No external model API or GPU is used. The first install needs access to the Python package registry.

For just viewing saved experiments and running offer predictions, this also works with only Python 3.10 or later:

```sh
python3 app.py
```

The CPU research option needs NumPy, SciPy and scikit-learn. Those are installed by `run.sh`. The browser calculates offer probabilities from the saved final weights, so opening the page does not retrain anything.

## The page

There are only two views. Experiments shows previous runs. New experiment lets you choose an offer prediction or the CPU research loop.

Offer runs are saved in this browser's IndexedDB before calculation, with the expectation, inputs, outputs and versions. Use Export my offer runs to keep a copy. These browser runs are not part of the research SQLite file and will not follow you to another device.

Model experiments are stored in `artifacts/runs.sqlite`. The local server lets you inspect them by ID and download a consistent database copy. `demo/runs.sqlite` and `demo/runs.json` are exported copies for a static version of the page. A static host can run the offer calculator, but it cannot run Python research; that option is only enabled when the local API is available.

## Commands I used

After installing the dependencies, these commands run the corresponding investigations:

```sh
.venv/bin/python experiments.py --show 2
.venv/bin/python regularization_test.py
.venv/bin/python frequency_baseline.py
.venv/bin/python structured_experiment.py
.venv/bin/python decision_experiment.py
.venv/bin/python research_loop.py
```

The first command reads run 2. The others append new saved experiments. They keep the originals. The research loop never loads check outcomes. It tries at most three settings per outer fold and stops after two changes fail to help enough. It does not automatically overwrite the final model.

`reproduce.py` repeats the original build and check comparisons. Its check results are reproduction of already-seen data, not a fresh test. It appends new runs. I would not use repeated check results to claim further validation.

To update the exported evidence without fitting a model:

```sh
.venv/bin/python build_demo.py
```

The demo stays tied to the final prototype saved in run 8. Updating the exports does not silently replace that model with a new research candidate.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -v
node tests/test_browser_model.js
```

Node is only needed for the Python/browser parity check. It is not needed to run the Python server or open the demo.

Source files matching the recorded model hashes are saved in `artifacts/sources/`. The failed run 1 source was reconstructed from the missing-value fix and checked against its previously recorded hash. I have kept that detail explicit instead of calling it an original saved snapshot.

`artifacts/environment.json` records the environment used to prepare the work. A different library version may give slightly different refits; the saved weights and predictions remain inspectable.

To recreate the submission ZIP, database copy, score exports and checksums without making a commit:

```sh
.venv/bin/python prepare_submission.py
```

The outputs go in `submission/`, which is ignored by Git. This does not send the submission or publish a live link.
