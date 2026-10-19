# Results I got after my experiments

My provisional choice to test is $3.49, an environmental message, online. I do not know the best launch offer yet. The model gives me a candidate, but real costs and real-shopper evidence are still missing.

## How the approach changed

I first checked if previous purchases help predict the next choice, especially buying versus skipping. Run 1 failed on blank spending fields, and I kept that failure. In run 2, history did not help. Even the starter performed better than the initial learned models.

So I thought maybe the learned effects were fitting noise. Run 5 used stronger regularization and improved both models. The small history advantage was still uncertain. 

Run 6 then showed that the inputs helped distinguish cleaners, but did not improve buying-versus-skipping predictions over average choice rates.

In run 7 I made a smaller model with 11 learned values. Its observed loss was similar to the earlier tuned model, and the explanation was easier to inspect. History was not selected.

## Scores

Lower log loss means better probability predictions. All rows below use build-data household cross-validation.

| Model | Log loss | Accuracy |
| --- | ---: | ---: |
| Average choice frequencies | 1.5971 | 25.3% |
| Original starter | 1.5793 | 29.0% |
| Initial current-only | 1.5909 | 28.8% |
| Initial with history | 1.5952 | 27.9% |
| Stronger regularization, current-only | 1.5731 | 29.4% |
| Stronger regularization, with history | 1.5706 | 29.5% |
| Final smaller model | 1.5694 | 30.7% |
| Smaller model with history | 1.5717 | 30.4% |

The final model's difference versus tuned current-only was -0.0037, with a descriptive household-bootstrap interval of about -0.0136 to +0.0055. That includes no improvement. I would not call this a proven accuracy breakthrough or anything. 

The same build folds were reused during development. Run 3 checked earlier models on 50 other households; the starter scored 1.5626 and initial current-only scored 1.5891 there. Those are not final-model scores. The final model has no new untouched evaluation.


 ## One Interesting thing I tried

Also since this was problem of multiuple experiements learning from it and then new experiment, I thought why cant i automate it, my inspiration was Andrej Karpathy's Auto research project (Although I know I cant just use that directly). So I made my own research loop. 

It tests predefined changes and saves expectations before running them. Run 9 kept the existing model after two changes failed to help enough in each fold. Its 15 fold-level trials include five baselines. Run 25 is an interface-verification repeat, not new evidence. This is not an unrestricted AI researcher or proof of the best possible model.

## The next real test

I would run one randomized online US shopper study comparing $3.49 versus $4.49 and cleaning versus a substantiated environmental message. Keep the formula, 16 fl oz bottle and competing offers constant. Include buying nothing, measure actual purchases and follow repeat purchases.

Get real costs first, then use a pilot to set the sample size and a worthwhile contribution-profit difference before the main study. Change the price if the discount does not pay for itself, or the message if cleaning performs better. Weak repeat purchase would also make me doubt the candidate. Online is a convenient test venue; superiority over retail remains untested.

## One more thing I tried

After I was ready to submit, I had a new idea. Why can't we just have a model tournament where we can try to find the best model? I did this, and I still found that our own model, which is a simple model, worked the best among the models I tried.
