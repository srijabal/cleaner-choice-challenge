"""Append replications, never overwrite original evidence or call the check set fresh."""
from experiments import read_run, register, execute

if __name__=='__main__':
    original=read_run(2)
    new=register('history_comparison','Reproduction of run 2 with already-known results. Expect the same negative history result; this is reproducibility verification, not a fresh hypothesis test.',original['inputs'])
    execute(new)
    chosen=read_run(new)['outputs']['selected_model']
    check=register('frozen_check','Reproduction of run 3 on the already-inspected check set. Expect the original results; this is not new out-of-sample evidence.',{'selection_run':new,'selected_model':chosen})
    execute(check)
