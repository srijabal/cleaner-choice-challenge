# Trying automatic research in a small way

One thing I wanted to try was the experiment loop from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). I adapted the pattern to a small CPU problem. This implementation chooses from predefined settings; it does not use an LLM to invent changes or rewrite code.

Run 9 sets up the loop. In each outer training fold, it uses three inner household folds to test a baseline and up to two proposed changes. It keeps a change only if inner loss improves by at least 0.002, buy/no-buy loss does not worsen by more than 0.005, and model checks pass. After two changes fail, it stops. Outer outcomes are only evaluated after that search finishes.

It kept the existing model in every fold. The 15 saved trials include five baselines and ten proposed changes. They are not 15 independent findings. Run 25 is the interface-verification repeat.

The code is `research_loop.py`. The check outcomes are never loaded by this loop. The build data have already been used in previous work, so nested validation here is still exploratory and does not establish a global optimum.

My reflection is in [AFTER_RESEARCH_LOOP.md](AFTER_RESEARCH_LOOP.md).
