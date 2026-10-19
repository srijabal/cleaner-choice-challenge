# What happened

The first experiment did not support the hypothesis: adding simple purchase history did not improve out-of-household predictions at the original regularization strength. The starter's hand-written rules also beat both learned models on log loss. The learned model is a transparent exploratory tool, not a demonstrated improvement over the starter.

## Approach

The two learned models are the same L2-regularized multinomial logistic regression (`C=1`, no tuning). They return probabilities for the four products and no purchase. Current features are prices, message (including missing), channel, survey budget/income/usual brand, bathrooms, and monthly cleaning spend. Prices are centered at $4; spending is divided by 20. These are fixed transforms, not estimates from test data. Blank spending uses a zero placeholder plus a missingness flag, not an assumption that spending was zero.

History adds availability of earlier visits, previous choice, whether the entrant was ever purchased, and trailing visits without a purchase. The last feature is a count of visits, not days or inferred bottle inventory. Survey notes, `said_why`, and `after_use_note` are unused. The history builder sorts each household's visits and adds the current outcome only after constructing that visit's input. It uses observed past purchases, not recursively simulated choices.

The chosen learned model carries **nothing between visits**. The rejected history model carries the state above. No household IDs are features. No outside data, LLM calls, or GPUs were used.

## Ledger

Expectations were saved in SQLite before executing each experiment. Completed and failed runs are retained.

- **Run 1:** failed on a blank spending field before producing evaluation metrics. The exception is saved.
- **Run 2:** five-fold build evaluation, split by household with seed 42 (160 train / 40 validation households in every fold). Added explicit missing-spending handling after run 1. Saved fold assignments, weights, per-visit probabilities, metrics, and paired household bootstrap intervals.
- **Run 3:** froze current-only after run 2, fit on all 200 build households, then evaluated the 50 check households once. All three models were compared as planned; no tuning or reselection followed. Later software tests load these same files but are not claimed as fresh model evaluations.
- **Run 4:** exploratory scenario sensitivity on 200 build households at their first visit, comparing observed entrant price/message/channel levels with competitors fixed at $3.99/$2.99/$4.29. Models were not refit.

All four original CSVs are unchanged. File SHA-256 hashes identify data versions; model source hashes and saved coefficients identify fitted models. Run 1 was registered before implementation and records its source hashes when execution began. Training used Python 3.14.7, NumPy 2.5.0 and scikit-learn 1.9.0; other compatible versions can produce small numerical differences.

## Prediction quality

Lower log loss is better: it penalizes confident wrong probabilities. Accuracy only counts the largest probability and can hide poor probability quality.

| Model | Build CV log loss | Check log loss | Check buy/no-buy loss | Check accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original starter | 1.5793 | 1.5626 | 0.4365 | 28.0% |
| Current-only | 1.5909 | 1.5891 | 0.4459 | 29.0% |
| With history | 1.5952 | 1.5959 | 0.4468 | 28.3% |

Run 2's history-minus-current loss difference was **+0.0044**, with a descriptive 95% household bootstrap interval **[-0.0071, +0.0144]**. Buy/no-buy loss also worsened, by +0.0043. The predeclared rule was to add history only if its loss difference and entire bootstrap interval were below zero. It was not selected.

Run 3's corresponding overall difference was **+0.0067**, interval **[-0.0108, +0.0251]**. This does not prove history can never help. It says these features, model and sample did not demonstrate a benefit. Bootstrap sampling uses whole households, but keeps fitted predictions fixed; it does not capture all training uncertainty, and CV folds share training observations.

The current-only model predicts entrant share of 20.3% on the check set versus 18.3% observed. For tight-budget households it predicts 18.9% versus 13.2% observed (144 visits, 24 households). Calibration bins, Brier scores, visit-number/channel/budget slices and all predictions are in run 3. Small slices are noisy; these are diagnostics, not independent confirmatory tests.

## Household explanations: evidence and doubt

Explanations are an exact decomposition of the predicted top choice's log odds versus the runner-up. All nonzero feature contributions, including the intercept, are shown. Correlated features can divide effects arbitrarily. These terms explain the calculation, not the person's true reasons.

Two deliberately selected check examples illustrate limits (not a representative success sample):

- **V201-4:** the first correctly predicted check visit. Standard gets 27.0%, budget 25.9%; the household bought standard. In standard-versus-budget log odds, usual standard brand contributes +0.438, while tight budget contributes -0.484. A one-line “usual brand” story would hide the competing influence and narrow margin. A correct top choice is weak evidence here.
- **V244-5:** the check visit with the largest current-only log loss. Budget gets 32.1% and standard 31.7%, but the household bought entrant, assigned only 6.1%. The available fields did not explain this outcome well. Do not invent a motivation to repair the story.

Tests verify temporal feature isolation, missing-spend handling, probability normalization, exact score decomposition, and matching Python/browser inference. These establish computational correctness, not shopper realism.

## Business decision and next test

**I do not know the best real-world price/message/channel yet.** Run 4 usually ranks **$3.49 / environment / online** highest by modeled purchase probability (current-only: 35.0%). One of five fold models instead prefers cleaning. The original starter also favors environment/online, but those preferences were manually encoded. Agreement with it is not independent corroboration.

The candidate has 67 observed build visits from repeated households; that count is not a randomized-effect estimate. Observed competitor offers vary, while scenario competitors are fixed. A lower price mechanically helps modeled demand but may hurt profit. Costs, margins, channel acquisition costs, realistic repurchase timing, randomization details and claim substantiation are absent. The fictional sample cannot establish external validity or causal effects. We did not test text features, richer history, nonlinear models, feature interactions, or other regularization strengths. That is a deliberate scope limit, not evidence against those ideas.

Propose **one online randomized 2×2 study** with real US shoppers: $3.49 vs $4.49, crossed with cleaning vs an independently substantiated environmental message. The provisional lead arm is $3.49/environment; online is chosen for a controllable test, not a proven channel advantage. Keep the formula, 16 fl oz bottle, competing offers and recruitment constant. Include no purchase and use actual purchase rather than stated intent. Follow participants for repeat purchase on a calendar schedule established through a pilot. Use truthful messages only; the supplied data do not validate any claim.

Before recruitment, obtain unit economics, define a minimum worthwhile contribution-profit difference, use a pilot to estimate baseline purchase rates, then set sample size and analysis before collecting the main outcomes. Randomize by shopper and account for repeated observations. Do not stop when a favorable result appears. Report confidence intervals for purchase, repeat purchase and contribution per exposed shopper. Prefer $4.49 if extra purchases at $3.49 do not cover the discount; prefer cleaning if it yields better prespecified outcomes. If intervals remain too wide, report uncertainty. A channel comparison needs a later test and remains unresolved.

## Follow-up: stronger regularization (run 5)

After seeing the original results, the next hypothesis was that the learned models were fitting noise. This is explicitly an exploratory follow-up chosen after the check results, not a fresh confirmation. No check outcomes were evaluated again.

For each of the same five outer household folds, three inner household folds selected C from 0.01, 0.1 and 1.0 using only the outer training households. Smaller C means a stronger penalty on large coefficients. Every outer fold selected C=0.01 for both feature sets. Outer validation outcomes were not used to choose C.

| Model | Original build CV loss | Nested-CV tuned loss |
| --- | ---: | ---: |
| Current-only | 1.5909 | 1.5731 |
| With history | 1.5952 | 1.5706 |
| Starter (unchanged) | 1.5793 | 1.5793 |

The tuned history model now has slightly lower loss than tuned current-only: difference -0.00246, descriptive household-bootstrap interval [-0.00537, +0.00033]. That interval still includes no improvement. Differences versus the starter also have intervals spanning zero. Stronger shrinkage improved the learned models relative to their original versions, but did not establish a reliable advantage over the starter or a reliable benefit from history. This is consistent with excessive flexibility being a problem; it does not prove the cause of the earlier errors.

The demo remains tied to the frozen run 3 model; run 5 is available in the updated downloadable ledger. Do not silently swap a tuned model into a previously reported check result. A next possible experiment is an intercept-only frequency baseline, to test whether the benefit mostly comes from predictions moving toward average choice rates. It has not been run.

## Review guide

Read run 2's expectation, then compare its results to FIRST.md. Explain why the hypothesis failed rather than proposing that “more AI” must fix it. For a new household, enter its survey information and offers in the demo, record an expectation, inspect probabilities and score contributions, and state the limits. Missing spending is allowed. Demo examples keep original prior history but fix competitor prices to the displayed defaults, so edited scenarios are not exact replay evaluations.
