# Giving the model less freedom

After the first comparison I thought maybe the learned effects were fitting noise. Run 5 tested stronger regularization. I used inner household folds to choose the setting and outer households to evaluate it, using only build data.

The stronger penalty helped both versions. Current-only loss went from 1.5909 to 1.5731. With history it went from 1.5952 to 1.5706. That small history advantage was still uncertain.

The code is `regularization_test.py`. My reflection is in [AFTER_EXPERIMENT_2.md](AFTER_EXPERIMENT_2.md). These were later comparisons on reused build data, not a new untouched test.
