import pandas as pd

from rta.misc import max_key_val



def choose_run(D, var2align, run):
    """Get input for the alignment.

    Do it by indicating a run to align to.

    Args:
        D (pd.DataFrame): DataFrame containing columns 'id', 'run', and ...
        var2align (str): Name of the column to align.
        run (whatever): The run to align to.
    Returns:
        tuple of pd.DataFrames: The data ready for alignment and the remainder.
    """
    X = D[['id', 'run', var2align]] # subselect the data for alignment
    X.columns = ['id', 'run', 'x'] 
    ref = X.loc[X.run == run] # the reference peptides
    other = X.loc[X.run != run] # all other peptides
    # we can align peptides in other runs only to those found in chosen run.
    alignable_idx = other.id.isin(set(other.id) & set(ref.id))
    X = other.loc[alignable_idx,]
    unalignable = other.loc[~alignable_idx,]
    ref = ref[['id','x']].set_index('id')
    ref.columns = ['y']
    X = pd.concat([X.set_index('id'), ref], axis=1, join='inner')
    return X, unalignable


def choose_most_shared_run(D, var2align, stats):
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
    X, unalignable = choose_run(D, var2align, most_shared_run)
    return X, unalignable


def stat_reference(X, stat='median', var='x', ref_name='y'):
    """Calculate the statistic summarizing X.

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


def choose_statistical_run(D, var2align, stat='median'):
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
    X = stat_reference(X, stat)
    unalignable = pd.DataFrame(columns=['id', 'run', 'x']) # empty DataFrame
    return X, unalignable