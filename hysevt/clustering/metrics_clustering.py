from sklearn import mixture, cluster
import numpy as np
import pandas as pd
import hysevt.clustering.evaluation


# Gaussian Mixture Model (finite)
def determine_cluster_number_GMM(X, n_components=np.arange(1, 21),gmm_kwargs=None):
    # from Python Data Science Handbook page 487
    models = [mixture.GaussianMixture(n, **gmm_kwargs).fit(X) for n in n_components]
    aic = np.array([m.aic(X) for m in models])
    bic = np.array([m.bic(X) for m in models])
    # add cluster scores
    out = [tuple(np.repeat(np.nan,4)),]
    for gmm in models[1:]:
        out.append(hysevt.clustering.evaluation.clustering_scores(X,gmm.predict(X)))
    results = pd.DataFrame(out,index=n_components,columns=["silhouette_score_cos","silhouette_score_eucl","calinski_harabasz_score","davies_bouldin_score"])
    results["AIC"] = aic
    results["BIC"] = bic
    results.index.name="n_clusters"
    return results,models


def best_n_cluster(results):
    try:
        best_n_cluster = dict(
            zip(
                [
                    "silhouette_score_cos",
                    "silhouette_score_eucl",
                    "calinski_harabasz_score",
                    "davies_bouldin_score",
                    "AIC",
                    "BIC",
                ],
                [
                    int(results.index[int(results.silhouette_score_cos.argmax())]),
                    int(results.index[int(results.silhouette_score_eucl.argmax())]),
                    int(results.index[int(results.calinski_harabasz_score.argmax())]),
                    int(results.index[int(results.davies_bouldin_score.argmin())]),
                    int(results.index[int(results.AIC.argmin())]),
                    int(results.index[results.BIC.argmin()]),
                ],
            )
        )
    except AttributeError:
        best_n_cluster = dict(
            zip(
                [
                    "silhouette_score_cos",
                    "silhouette_score_eucl",
                    "calinski_harabasz_score",
                    "davies_bouldin_score",
                ],
                [
                    int(results.index[int(results.silhouette_score_cos.argmax())]),
                    int(results.index[int(results.silhouette_score_eucl.argmax())]),
                    int(results.index[int(results.calinski_harabasz_score.argmax())]),
                    int(results.index[int(results.davies_bouldin_score.argmin())]),
                ],
            )
        )
    return best_n_cluster

# agglomerative clustering
def determine_cluster_number_agglomerative(X, n_clusters=np.arange(1, 21),agl_kwargs=None):
    n_clusters = np.arange(2, 21)
    models = [
        cluster.AgglomerativeClustering(n, **agl_kwargs).fit(X)
        for n in n_clusters
    ]
    scores = [
        hysevt.clustering.evaluation.clustering_scores(
            X, model.fit_predict(X)
        )
        for model in models
    ]
    results = pd.DataFrame(
        np.array(scores),
        index=n_clusters,
        columns=[
            "silhouette_score_cos",
            "silhouette_score_eucl",
            "calinski_harabasz_score",
            "davies_bouldin_score",
        ],
    )
    results.index.name="n_clusters"
    return results,models

def AGL_estimate_dist_threshold_from_quantile(X,q):
    agglCluster = cluster.AgglomerativeClustering(compute_distances=True).fit(
        X
    )
    return np.quantile(agglCluster.distances_, q),agglCluster.distances_,agglCluster
