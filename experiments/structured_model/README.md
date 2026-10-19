# Making the final model smaller

In run 7 I tried a shared-utility model with 11 learned values instead of separate effects for every choice. It uses product baselines, price, tight-budget price sensitivity, usual brand, entrant message and entrant channel.

Its build loss was 1.5694. The history version scored 1.5717, so history was not selected. The small difference versus the earlier tuned current-only model was uncertain. I chose this model because its structure and explanations were easier to inspect, while the observed prediction quality was similar.

Higher prices cannot increase own purchase probability in this model. That is an imposed assumption. Its explanations add up to the score difference, but they do not establish a shopper's motivation.

The code is `structured_model.py` and `structured_experiment.py`. My reflection is in [AFTER_EXPERIMENT_3.md](AFTER_EXPERIMENT_3.md).
