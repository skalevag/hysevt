from sklearn import metrics
import scipy
from functools import partial

# cluster evaluation metrics
def clustering_scores(X,y):

    cluster_metric_functions = [
        partial(metrics.silhouette_score, metric="cosine"),
        partial(metrics.silhouette_score, metric="euclidean"),
        metrics.calinski_harabasz_score,
        metrics.davies_bouldin_score,
    ]

    return tuple([func(X, y) for func in cluster_metric_functions])


def calc_metric_zscores(
    metrics,
    cluster_labels,
    drop_cols=[
        "start",
        "end",
    ],
):
    # calculate zscores
    metric_zscores = metrics.drop(columns=drop_cols).apply(
        partial(scipy.stats.zscore, nan_policy="omit")
    )
    # remove some columns and add cluster labels
    metric_zscores = metric_zscores.join(cluster_labels)  # add cluster labels
    return metric_zscores


def calc_cluster_zscore_averages(
    metric_zscores, mode="median", cluster_id_col="cluster_id"
):
    if mode == "median":
        metric_zscore_averages = metric_zscores.groupby(cluster_id_col).median()
    elif mode == "mean":
        metric_zscore_averages = metric_zscores.groupby(cluster_id_col).mean()
    else:
        raise ValueError("mode must be either 'median' or 'mean'")
    return metric_zscore_averages

def calc_cluster_zscore_error_range(
    metric_zscores, error_range="full", cluster_id_col="cluster_id"
):
    if error_range == "full":
        q_min, q_max = (0, 1)
    elif type(error_range) is tuple:
        q_min, q_max = error_range
    else:
        raise ValueError(
            "error_range must be a tuple of quantiles, or 'full' if whole range is wanted"
        )
    return metric_zscores.groupby(cluster_id_col).quantile(
        q_min
    ), metric_zscores.groupby(cluster_id_col).quantile(q_max)