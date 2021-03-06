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
    return np.array([[np.sum(real &  pred), np.sum(~real &  pred)],
                     [np.sum(real & ~pred), np.sum(~real & ~pred)]], 
                     dtype=int)


def accuracy(cm):
    """True positives and true negatives vs all assignments.

    Args:
        cm (2x2 np.array): A confusion matrix.
    Returns:
        float: Accuracy of the classifier.
    """
    return (cm[0,0] + cm[1,1])/np.sum(cm)


def sensitivity(cm):
    """True positives vs all positives.

    Args:
        cm (2x2 np.array): A confusion matrix.
    Returns:
        float: Sensitivity of the classifier.
    """
    return cm[0,0]/(cm[0,0] + cm[0,1])


def false_discovery_rate(cm):
    """False positives vs all positives.

    Args:
        cm (2x2 np.array): A confusion matrix.
    Returns:
        float: False discovery rate of the classifier.
    """
    return cm[0,1]/(cm[0,0] + cm[0,1])


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
    r_perc = dict(run_data.rt.apply(np.percentile, q=quantiles))

    def iter_distances_from_fold_perc_2_real_perc():
        for (run, fold), rf_d in data.groupby([run_name, fold_name]):
            out = np.array((run, fold))
            rf_perc = np.percentile(rf_d.rt, q=quantiles)
            out = np.append(out, np.abs(rf_perc - r_perc[run]))
            yield out

    distances = pd.DataFrame(iter_distances_from_fold_perc_2_real_perc())
    distances[[0,1]] = distances[[0,1]].astype(int)
    cols = [run_name, fold_name] + list(quantiles)
    distances.columns = cols
    distances.set_index([run_name, fold_name], inplace=True)
    distances_run = pd.DataFrame(distances.groupby(run_name).describe())
    distances_agg = pd.DataFrame(distances.describe())
    return distances, distances_run, distances_agg



def __describe_runs(data,
                    var_name,
                    q=np.arange(0, 101, 5)):
    """Basic description of the data set.

    Number of peptides per run.
    """
    data_run = data.groupby('run')
    out = pd.DataFrame(data_run.rt.count())
    out.columns=['count']
    out = pd.concat([out,
                     pd.DataFrame(data_run[var_name].apply(np.percentile,
                                                           q=q).values.tolist(),
                                  columns=q,
                                  index=out.index)],
                    axis=1)
    return out



def describe_runs(data,
                  var_names=['rt', 'dt', 'mass'],
                  quantiles=np.arange(0, 101, 5)):
    """Basic description of the data set.

    Args:
        data (pandas.DataFrame): DataFrame with column 'run' and other features.
        var_names (string or list of strings): names of features to report on.
        quantiles (iterable): which quantiles should be compared, in range 0-100.

    Return:
        out (dictionary): a dictionary with run summaries and percentile of feature distribution per run.
    """
    v = var_names if isinstance(var_names, list) else [var_names]
    return {n:__describe_runs(data, n, quantiles) for n in v}



def max_space(x, sort=False):
    """Calculate the maximal space between a set of 1D points.

    If there is only one point, return 0.

    Args:
        x (np.array) The array for which we have to find the maximal space.
    """
    if len(x) == 1:
        return 0
    else:
        if sort:
            return np.diff(np.sort(x)).max()
        else:
            return np.diff(x).max()