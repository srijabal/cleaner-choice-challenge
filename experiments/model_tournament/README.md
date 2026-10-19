# Model tournament

After I was ready to submit, I had a new idea. Why can't we just have a model tournament where we can try to find the best model? I did this, and I still found that our own model, which is a simple model, worked the best among the models I tried.

I tried 17 settings. The search chose the current-only utility model in all 5 outer folds. This was still using the same build data, so it does not prove that this is the best possible model.

| Model | Log loss, lower is better |
| --- | --- |
| Shared utility | 1.5694 |
| Boosted trees | 1.5775 |
| Logistic regression | 1.5790 |
| Random forest | 1.5802 |

- Code: [model_tournament.py](../../model_tournament.py)
- Results: [tournament_summary.json](../../artifacts/tournament_summary.json)
- Saved experiment: run 41; its 85 trial records are runs 42–126 in [the database](../../artifacts/runs.sqlite). These are settings across folds, not 85 independent experiments.
- Full run export: [41.json](../../demo/runs/41.json)
- Run again: `.venv/bin/python model_tournament.py`
