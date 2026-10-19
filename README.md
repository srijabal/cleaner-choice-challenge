# Cleaner choice

I started this by checking if remembering previous purchases actually helps predict the next choice. It did not help in the first model. After that I tried stronger regularization, compared against average choice rates, and made a smaller model that is easier to explain.

The final prototype has 11 learned values. My provisional test choice is $3.49, environmental messaging, online. This is a starting point for a real shopper test. I do not know the best launch offer yet, especially because actual costs are missing.

## Run it

```sh
./run.sh
```

Then open http://127.0.0.1:8000. The first run installs the CPU dependencies in a local virtual environment. No model API or GPU is needed.

If you only want to inspect saved experiments and try offers, `python3 app.py` also works without installing training dependencies. The CPU research option needs the dependencies.

The page only has Experiments and New experiment. You can open a saved run by ID, change an offer, or start a short automatic research loop. Offer predictions are saved in the browser and can be exported. Model experiments are saved in SQLite.

## What to read

- [FIRST.md](FIRST.md) is my initial concern, committed before code changes.
- [Experiment notes](experiments/README.md) explain what I tested and what changed my mind.
- [RESULTS.md](RESULTS.md) has the final model, scores, limits and proposed shopper test.
- [FINAL.md](FINAL.md) is the short final note.
- [RUNNING.md](RUNNING.md) has the commands and database details.
- [ASSIGNMENT.md](ASSIGNMENT.md) keeps the original instructions unchanged.

The run database is [artifacts/runs.sqlite](artifacts/runs.sqlite). The original CSV files and starter Python files are unchanged. All supplied households and product claims are fictional.
