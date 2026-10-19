# Does the household split change the result?

This is a follow-up after submission. I wanted to check if I got lucky with the household split. So I tried 5 different splits, keeping all visits from a household together. I did not change the model settings.

The simple model still did better than average choice rates for predicting which cleaner gets chosen. History was a tiny bit better in one split and worse in the other four. So I still cannot say that history never matters. Also the simple model did not do better than average choice rates for predicting buy or skip.

| Model | Mean log loss | Mean buy/skip loss |
| --- | --- | --- |
| Current-only utility | 1.5691 | 0.4591 |
| Utility with history | 1.5703 | 0.4598 |
| Regularized logistic | 1.5735 | 0.4572 |
| Average choice rates | 1.5989 | 0.4581 |

Lower loss is better. The utility model's choice loss ranged from 1.5666 to 1.5721 across the splits. This range is not a confidence interval. Its small advantage over logistic is still uncertain.

These are the same 200 households again, so this checks sensitivity to the split, not performance on new shoppers. The submitted model and final note are unchanged.

- Saved experiment: run 127; runs 128–132 hold the five split checks.
- [Code](../../split_robustness.py)
- [Results and every split's scores](../../artifacts/split_robustness_summary.json)
- [Full saved run](../../demo/runs/127.json)
- Run again: `.venv/bin/python split_robustness.py`
