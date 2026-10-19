# Turning predictions into an offer to test

Run 4 first compared the observed price, message and channel options using the earlier models. Most fits preferred $3.49/environment/online, but one fold preferred cleaning. I kept that disagreement.

Run 8 repeated the scenario comparison with the smaller final model. All five utility fold fits preferred $3.49/environment/online for purchase share. These are predictions for a fixed set of competitor prices, not measured effects of changing an offer.

The cost comparison was important. The price with the most modeled purchases was not always the price with the most contribution per exposure. Hypothetical costs changed the preferred price, so actual unit economics are needed before a launch decision.

The code is `scenario_test.py` for the earlier comparison and `decision_experiment.py` for the final one. The full recommendation and real-shopper test are in [RESULTS.md](../../RESULTS.md).
