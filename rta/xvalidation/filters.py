import numpy as np
from collections import Counter as count

def filter_foldable(annotated, 
                    annotated_stats, 
                    K):
    """Filter peptides divisible into K cv-folds."""
    runs_no = max(annotated_stats.runs_no)
    run_participation_cnt = count(annotated_stats.runs)
    run_participation_cnt = sorted(run_participation_cnt.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
    runs, peptides_cnt = map(np.array, zip(*run_participation_cnt))
    foldable_runs = set(runs[peptides_cnt >= K])
    annotated_cv = annotated[ annotated.runs.isin(foldable_runs) ]
    return annotated_cv