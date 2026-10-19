# First question: does history help?

I wanted to check if previous purchases help predict the next choice, especially whether the household buys or skips. I compared the same general model with and without history, keeping whole households together in validation.

Run 1 failed because some spending fields were blank. I fixed the missing-value handling and kept the failure. Run 2 did not show a benefit from adding history. Run 3 checked the earlier models on the other households, and the starter was still competitive.

This does not establish that history never matters. It tells me these features and this model did not demonstrate a useful improvement.

My original reflection is in [AFTER_EXPERIMENT_1.md](AFTER_EXPERIMENT_1.md). The code is in `choice_model.py` and `experiments.py`.
