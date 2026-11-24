"""Replay a validation sample from the newly cloned Modal training oracles on Kaggle."""

from dump_oracles import PAIRS, SMOKE_INPUT, SMOKE_RESULT
from stage1_smoke import kaggle_replay


if __name__ == "__main__":
    results = kaggle_replay(input_name=SMOKE_INPUT, result_name=SMOKE_RESULT, validation=True)
    if results.keys() != PAIRS.keys():
        raise ValueError("Expected validation replay for P1, P2, and P4.")
