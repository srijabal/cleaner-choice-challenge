# How the experiments went

To keep things organized i decided to keep the main ideas in separate folders. A saved run is not always a new independent experiment. Some runs are failures, validation folds, or repeats to check the interface.

| Question | Saved runs | Notes |
| --- | --- | --- |
| Does previous purchase history help? | 1 failed, 2 build comparison, 3 earlier-model check | [History](history/README.md) |
| Which offer does a model prefer? | 4 exploratory, 8 final-model scenarios | [Offers](offer_scenarios/README.md) |
| Are the learned effects too flexible? | 5 | [Regularization](regularization/README.md) |
| Do the inputs beat average choice rates? | 6 | [Frequency comparison](frequency_baseline/README.md) |
| Can a smaller model explain offers clearly? | 7 | [Smaller model](structured_model/README.md) |
| Can a bounded loop improve it further?(Auto Research Inpsired)  | 9, with child trials 10–24 | [Research loop](research_loop/README.md) |
| Does trying more model families help? | 41, with child trials 42–126 | [Model tournament](model_tournament/README.md) |
| Does changing the household split change the result? (after submission) | 127, with split checks 128–132 | [Split sensitivity](split_robustness/README.md) |

Run 25 and its child trials just repeat the loop to verify the interface. 

Expectations recorded before execution, inputs, outputs, failures and versions are in `artifacts/runs.sqlite`. I have tried to writed a note after every experiment to show my thought process. 