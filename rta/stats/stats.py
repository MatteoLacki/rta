import numpy as np
import pandas as pd


def l1(x, y):
    """Calculate the sum of absolute differences of entries of 'x' and 'y'."""
    return np.abs(x-y).sum()



def mae(x):
    """Calculate the mean absolute deviation.

    Args:
        x (np.array): preferably floats.

    Returns:
        out (float): the mean absolute deviation for array 'x'.
    """
    return np.mean(np.abs(x))



def mad(x, return_median=False):
    """Compute median absolute deviation from median.

    Args:
        x (np.array): preferably floats.
        return_median (boolean): should we return also the median?

    Return:
        out (float, or tupple): median absolute deviation for "x", potentially together with its median.
    """
    median = np.median(x)
    if return_median:
        return np.median(np.abs(x - median)), median
    else:
        return np.median(np.abs(x - median))



def confusion_matrix(real, pred):
    """A confusion matrix that is quicker than SKLearn one.
    
    Args:
        real (np.array, dtype=boolean): real outcomes.
        real (np.array, dtype=boolean): predicted outcomes.

    Return:
        out (np.array): confusion matrix.
    """
    return np.array([[np.sum(np.logical_and( real, pred)), np.sum(np.logical_and(~real, pred))],
                     [np.sum(np.logical_and( real,~pred)), np.sum(np.logical_and(~real,~pred))]], 
                     dtype=int)



def compare_fold_quantiles(data,
                           quantiles = np.arange(0, 101, 5),
                           param_name = "rt",
                           run_name = "run",
                           fold_name = "fold"):
    """Compare quantiles across folds and with the data itself.

    A tool to check if the folds are representative in terms of
    the distributions of the given parameter.

    Args:
        data (pandas.DataFrame): Data with runs and folds and a feature to compare between folds.
        quantiles (iterable): which quantiles should be compared, in range 0-100.
        param_name (str): name of the column with the feature.
        run_name (str): name of the column with runs.
        fold_name (str): name of the column with folds.

    Return:
        real_perc (numpy.array): percentiles of the feature in real data.
        fold_perc (pandas.DataFrame): percentiles of the feature per fold and run.
        fold_perc_stats_by_run (pandas.DataFrame): percentiles of the feature per run.
        fold_perc_stats (pandas.DataFrame): aggregated percentiles of the feature.
    """
    run_fold_data = data.groupby([run_name, fold_name])
    fold_perc = run_fold_data[param_name].apply(np.percentile, 
                                       q=quantiles)
    fold_perc = pd.DataFrame(fold_perc.values.tolist(),
                             columns=quantiles,
                             index=fold_perc.index)
    fold_perc_stats = fold_perc.describe()
    fold_perc_stats_by_run = fold_perc.groupby('run').describe()
    real_perc = np.percentile(data[param_name], q=quantiles)
    return real_perc, fold_perc, fold_perc_stats_by_run, fold_perc_stats



def fold_similarity(data, 
                    quantiles = np.arange(0, 101, 5),
                    param_name = "rt",
                    run_name = "run",
                    fold_name = "fold"):
    """Evaluate the similarity between percentiles of folds and actual data.

    Args:
        data (pandas.DataFrame): Data with runs and folds and a feature to compare between folds.
        quantiles (iterable): which quantiles should be compared, in range 0-100.
        param_name (str): name of the column with the feature.
        run_name (str): name of the column with runs.
        fold_name (str): name of the column with folds.

    Return:
        distances (pandas.DataFrame): distances of folds per run to per run data.
        distances_run (pandas.DataFrame): distances aggregated over folds.
        distances_agg (pandas.DataFrame): distances aggregated over folds and runs.
    """
    run_data = data.groupby(run_name)
    r_perc = dict(run_data.rt.apply(np.percentile, q=q))

    def iter_distances_from_fold_perc_2_real_perc():
        for (run, fold), rf_d in data.groupby([run_name, fold_name]):
            out = np.array((run, fold))
            rf_perc = np.percentile(rf_d.rt, q=q)
            out = np.append(out, np.abs(rf_perc - r_perc[run]))
            yield out

    distances = pd.DataFrame(iter_distances_from_fold_perc_2_real_perc())
    distances[[0,1]] = distances[[0,1]].astype(int)
    cols = [run_name, fold_name] + list(q)
    distances.columns = cols
    distances.set_index([run_name, fold_name], inplace=True)
    distances_run = pd.DataFrame(distances.groupby(run_name).describe())
    distances_agg = pd.DataFrame(distances.describe())
    return distances, distances_run, distances_agg
