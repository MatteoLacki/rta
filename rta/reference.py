import numpy as np
import pandas as pd

from rta.misc import max_key_val



def cond_medians(x, g):
    """Get Me(x|g) cast over all entries of x.
    
    Args:
        x: inputs
        g: groups
    Returns:
        np.array(shape=x.shape): medians
    """
    DF = pd.DataFrame(dict(x=x, id=g)).set_index('id')
    W = DF.groupby('id').median().rename(columns={'x':'y'})
    y = pd.concat([DF, W], join='inner', axis=1).y.values
    unalignable = np.array([])
    return y, unalignable


def _choose_run(X, j):
    ref = X.loc[X.run == j] # the reference peptides
    other = X.loc[X.run != j] # all other peptides    
    alignable_idx = other.id.isin(set(other.id) & set(ref.id))
    X = other.loc[alignable_idx,]
    unalignable_peptides = other.id[~alignable_idx].unique()
    ref = ref[['id','x']].set_index('id')
    ref.columns = ['y']
    X = pd.concat([X.set_index('id'), ref], axis=1, join='inner')
    return X, unalignable_peptides


def choose_run(x, pept_id, run, j):
    X = pd.DataFrame({'id':pept_id, 'x':x, 'run':run})
    X, unalignable_peptides = _choose_run(X, j)
    return X.y.values, unalignable_peptides


def cond_medians_pd(D, var2align, j):
    """Get input for the alignment.

    Do it by indicating a run to align to.

    Args:
        D (pd.DataFrame): DataFrame containing columns 'id', 'run', and ...
        var2align (str): Name of the column to align.
        j (whatever): The run to align to.
    Returns:
        tuple of pd.DataFrames: The data ready for alignment and the remainder.
    """
    X = D[['id', 'run', var2align]] # subselect the data for alignment
    X.columns = ['id', 'run', 'x']
    X, unalignable_peptides = _choose_run(X, j)
    return X, unalignable_peptides


def choose_most_shared_run_pd(D, var2align, stats):
    """Get input for the alignment.

    Do it by choosing the run having most shared peptides with other runs.

    Args:
        D (pd.DataFrame): DataFrame containing columns 'id', 'run', and ...
        var2align (str): Name of the column to align.
        run (whatever): The run to align to.
    Returns:
        tuple of pd.DataFrames: The data ready for alignment and the remainder.
    """
    runs = D.run.unique()
    peptide_shares = {r: sum(len(p_runs)-1 for p_runs in stats.runs if r in p_runs) for r in runs}
    most_shared_run, shares = max_key_val(peptide_shares)
    X, unalignable = choose_run_pd(D, var2align, most_shared_run)
    return X, unalignable


def stat_reference_pd(X, stat='median', var='x', ref_name='y'):
    """Calculate the statistic summarizing pd.DataFrame X.

    Args:
        X (pd.DataFrame): DataFrame with columns 'id' and 'x'.
        stat ('median' or 'mean'): The statistic to calculate.
        ref_name (str): Name given to the output column.
    Return:
        X with and appended column with the reference."""
    assert stat in ('median', 'mean')
    if stat=='median':
        ref = X.groupby('id')[var].median()
    else:
        ref = X.groupby('id')[var].mean()
    ref.name = ref_name
    X = pd.concat([X, ref], axis=1, join='inner')
    return X


def choose_statistical_run_pd(D, var2align, stat='median'):
    """Get input for the alignment.

    Do it by choosing the median run to compare to.

    Args:
        D (pd.DataFrame): DataFrame containing columns 'id', 'run', and ...
        var2align (str): Name of the column to align.
    Returns:
        tuple of pd.DataFrames: The data ready for alignment and the empty remainder (for compatibility with other reference selectors).
    """
    X = D[['id', 'run', var2align]]
    X.columns = ['id', 'run', 'x']
    X.set_index('id', inplace=True)
    X = stat_reference_pd(X, stat)
    unalignable = pd.DataFrame(columns=['id', 'run', 'x']) # empty DataFrame
    return X, unalignable