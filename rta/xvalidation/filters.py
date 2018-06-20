import numpy as np
from collections import Counter as count

def filter_K_foldable(annotated, annotated_stats, K):
    """Filter peptides divisible into K cv-folds.

    Returns sparser copies of the original DF and its statistics.
    """
    run_counts = count(annotated_stats.runs)
    infrequent_runs = set(el for el, cnt in run_counts.items() if cnt < K)
    annotated_cv = annotated[~annotated.runs.isin(infrequent_runs)]
    annotated_stats_cv = annotated_stats[~annotated_stats.runs.isin(infrequent_runs)]
    return annotated_cv, annotated_stats_cv, run_counts
