# Are the inputs adding anything?

In run 6 I compared the models with a basic baseline that gives everyone the same probabilities, based on choice frequencies in the training households. I estimated those probabilities separately in each fold, without using its held-out outcomes.

The frequency baseline had log loss 1.5971. The tuned current-only model had 1.5731 and the history version had 1.5706. So the inputs were doing something useful overall.

But there was an interesting limit. The improvement came from distinguishing cleaners when someone bought something. The models still did not improve buying-versus-skipping loss over the frequency baseline. That made me more careful about the original purchase-timing idea.

The code is `frequency_baseline.py`. This reused saved build predictions; it did not evaluate check outcomes again.
