"""
Plotting functions for visualising water and sediment data.

author:
Amalie Skålevåg
skalevag2@uni-potsdam.de
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import datetime as dt
import hysevt.events.metrics
from scipy.cluster.hierarchy import dendrogram
import scipy
import matplotlib as mpl

def plot_years_together(
    ts, y_column, months=range(5, 11), ylabel=None,
):
    fig, ax = plt.subplots(figsize=(20, 6))

    custom_cycler = cycler(
        color=plt.cm.rainbow(np.linspace(0, 1, len(ts.year.unique())))
    )
    ax.set_prop_cycle(custom_cycler)
    y_upper = 0

    for label, df in ts.groupby("year"):
        try:
            freq = pd.infer_freq(df.index)
        except ValueError:
            continue
        idx = pd.period_range(
            dt.datetime(min(df.index.year), 1, 1, 0, 0, 0),
            dt.datetime(max(df.index.year), 12, 31, 0, 0, 0),
            freq=freq,
        ).to_timestamp()
        df = df.reindex(idx)
        y_upper = max(
            y_upper,
            np.nanmax(
                df.loc[f"{label}-{months[0]:02d}":f"{label}-{months[-2]:02d}"][y_column]
            ),
        )
        ax.plot(df[y_column].values, label=label)
    plt.legend()
    if ylabel is None:
        ax.set_ylabel(y_column)
    else:
        ax.set_ylabel(ylabel)
    lab = dict(
        zip(
            [np.where(df.index == df.month.eq(m).idxmax())[0][0] for m in months],
            [dt.date(2000, m, 1).strftime("%b") for m in months],
        )
    )
    ax.set_ylim(0, y_upper + y_upper * 0.05)
    ax.set_xlim(min(list(lab.keys())), max(list(lab.keys())))
    ax.set_xticks(list(lab.keys()))
    ax.set_xticklabels(lab.values())

    return fig

def plotEventHysteresis(event_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        event_data.streamflow,
        event_data.suspended_sediment,
        "-",
        color="grey",
        zorder=0,
    )
    event_data.plot.scatter(
        x="streamflow",
        y="suspended_sediment",
        c=range(len(event_data)),
        cmap="copper",
        ax=ax,
        s=30,
    )
    ax.set_title("Hysteresis")
    ax.annotate(event_data.index[0], event_data.iloc[0, :2].values[::-1])
    ax.annotate(event_data.index[-1], event_data.iloc[-1, :2].values[::-1])
    ax.set_ylabel("SSC [mg/l]")
    ax.set_xlabel("Q [m3/s]")

def plotEventSeries(event_data, ax=None, legend=False):
    if ax is None:
        fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(event_data.streamflow, "b", label="streamflow")
    ax.plot(hysevt.events.metrics.get_streamflow_peaks(event_data.streamflow),"kx")
    ax.set_ylabel("Q [m3/s]",color="b")
    ax.tick_params(axis='y', labelcolor="b")
    ax2.plot(event_data.suspended_sediment, "brown", label="suspended_sediment")
    ax2.plot(hysevt.events.metrics.get_suspended_sediment_peaks(event_data.suspended_sediment),"kx")
    ax2.set_ylabel("SSC [mg/l]",color="brown")
    ax2.tick_params(axis='y', labelcolor="brown")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if legend:
        ax.legend(h1 + h2, l1 + l2)
    ax.set_title("Event")
    
def plotEventSeriesWithPrecip(
    event_data,
    pre_event_data,
    event_precip,
    legend=True,
    precip_axis_scaling_factor=2.5,
    gauge_axis_scaling_factor=1.5,
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))

    ax2 = ax.twinx()
    ax3 = ax.twinx()

    # plot streamflow
    ax.plot(event_data.streamflow, "b", label="Q during event")
    ax.plot(pre_event_data.streamflow, "b:", label="Q before event")
    ax.plot(
        hysevt.events.metrics.get_streamflow_peaks(event_data.streamflow),
        "kx",
    )
    ax.set_ylabel("Streamflow (Q) [$m^3\ s^{-1}$]", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * gauge_axis_scaling_factor)

    # plot suspended sediment
    ax2.plot(event_data.suspended_sediment, "brown", label="SSC during event")
    ax2.plot(
        pre_event_data.suspended_sediment, ":", color="brown", label="SSC before event"
    )
    ax2.plot(
        hysevt.events.metrics.get_suspended_sediment_peaks(
            event_data.suspended_sediment
        ),
        "kx",
    )
    ax2.set_ylabel("Suspended sediment concentration (SSC) [$mg\ l^{-1}$]", color="brown")
    ax2.tick_params(axis="y", labelcolor="brown")
    ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] * gauge_axis_scaling_factor)

    # plot precipitation
    ax3.bar(
        x=event_precip.index,
        height=event_precip.RR,
        width=0.03,
        color="grey",
        label="P before and during event",
    )
    ax3.set_ylim(np.array(ax3.get_ylim()[::-1]) * precip_axis_scaling_factor)
    ax3.spines.right.set_position(("axes", 1.2))
    ax3.set_ylabel("Precipitation (P) [$mm\ h^{-1}$]", color="grey")
    ax3.tick_params(axis="y", labelcolor="grey")

    # legend handles
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    if legend:
        ax.legend(h1 + h2 + h3, l1 + l2 + l3, bbox_to_anchor=(1.3, 1), loc="upper left")
    ax.set_title("Event")
    

def plot_MultivariateKmedoids_eval(results,ODIR=None):
    fig, ax = plt.subplots(nrows=2, sharex=False, figsize=(5, 5))
    results.plot(x="n_clusters", y="SSE", ax=ax[0], style=".-", grid=True)
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Sum of squared errors (SSE)")
    results.plot.bar(x="n_clusters", y="n_iterations", ax=ax[1])
    ax[1].set_xlabel("number of clusters")
    ax[1].set_ylabel("number of iterations")
    ax[1].set_yticks(np.arange(ax[1].get_yticks().min(), ax[1].get_yticks().max(), 2))
    plt.savefig(ODIR.joinpath("METS_clustering_results_eval.png"),bbox_inches="tight")

def eval_plot_PCA(pca,cum_var_level=0.9):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(np.arange(1, pca.n_components_ + 1), pca.explained_variance_)
    ax1.set_xticks(np.arange(1, pca.n_components_ + 1))
    ax2.plot(
        np.arange(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), "r-*"
    )
    ax2.hlines(cum_var_level, 0, pca.n_components_ + 1, "k")
    ax1.set_ylabel("Eigenvalue (Explained variance)")
    ax2.set_ylabel("Cumulative explained variance ratio")
    ax1.set_xlabel("Component")
    return fig

def plot_variables_in_pca_space(loadings,variables,c1,c2,legend=True,ax=None,colors = None):
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(variables)))
    for i, feature in enumerate(variables):
        x = loadings[i, c1 - 1]
        y = loadings[i, c2 - 1]
        ax.plot(
            [0, x],
            [0, y],
            color=colors[i],
            label=feature,
        )
        ax.annotate(text=variables[i],
                    xy = (x,y),
                    horizontalalignment="right" if x < 0 else "left",
                    verticalalignment="bottom" if y > 0 else "top",color=colors[i])
    
    ax.set_xlabel(f"Component {c1}")
    ax.set_ylabel(f"Component {c2}")
    if legend:
        ax.legend(bbox_to_anchor=(1, 1))

def plot_loadings_for_component(loadings, feature_labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.barh(y=np.arange(len(feature_labels)), width=loadings)
    ax.set_yticks(ticks=np.arange(len(feature_labels)), labels=feature_labels)

def quickplot_clustering_results(clustering_results: pd.DataFrame, with_year_column=False):
    """Plots the event metrics grouped by cluster results.

    Args:
        clustering_results (pd.DataFrame): event_metrics dataframe with cluster labels in column "cluster_id"
        with_year_column (bool, optional): whether or not to include the year. Defaults to False.
    """
    # get events that were not assigned a cluster
    null_cluster = clustering_results[clustering_results.cluster_id == -1]
    # remove those events from the cluster results dataframe
    clustering_results = clustering_results[clustering_results.cluster_id != -1]

    fig_list = []
    # number of events in each cluster
    fig, ax = plt.subplots()
    numbers = clustering_results.groupby("cluster_id").cluster_id.count()
    numbers.index = numbers.index.astype(int)
    numbers.plot.bar(ax=ax)
    plt.ylabel("Number of events in cluster")
    fig_list.append(fig)

    # scatter plots colored by clusters
    # discharge vs. sediment yield
    fig, ax = plt.subplots()
    null_cluster.plot.scatter(ax=ax, y="SSY", x="Qtotal", color="grey", marker="x")
    clustering_results.plot.scatter(
        ax=ax,
        y="SSY",
        x="Qtotal",
        c=clustering_results["cluster_id"],
        cmap="viridis",
        colorbar=True,
    )
    fig_list.append(fig)

    # seasonality vs. sediment yield
    fig, ax = plt.subplots()
    null_cluster.plot.scatter(
        ax=ax, y="SSY", x="seasonal_timing", color="grey", marker="x"
    )
    clustering_results.plot.scatter(
        ax=ax,
        y="SSY",
        x="seasonal_timing",
        c=clustering_results["cluster_id"],
        cmap="viridis",
        colorbar=True,
    )
    fig_list.append(fig)

    # sediment related
    fig, ax = plt.subplots()
    clustering_results.boxplot(
        column=["SSY", "SSC_max"], by="cluster_id", grid=False, color="brown",ax=ax
    )
    plt.tight_layout()
    fig_list.append(fig)

    # streamflow related
    fig, ax = plt.subplots(ncols=2)
    clustering_results.boxplot(
        column="Qtotal", by="cluster_id", grid=False, color="blue", ax=ax[0]
    )
    clustering_results.boxplot(
        column="Q_max", by="cluster_id", grid=False, color="blue", ax=ax[1]
    )
    plt.tight_layout()
    fig_list.append(fig)

    # hysteresis
    fig, ax = plt.subplots(ncols=3)
    clustering_results.boxplot(
        column="SHI", by="cluster_id", grid=False, color="green", ax=ax[0]
    )
    clustering_results.boxplot(
        column="AHI", by="cluster_id", grid=False, color="green", ax=ax[1]
    )
    clustering_results.boxplot(
        column="peak_phase_diff",
        by="cluster_id",
        grid=False,
        color="green",
        ax=ax[2],
    )
    plt.tight_layout()
    fig_list.append(fig)

    # complexity
    fig, ax = plt.subplots(ncols=2)
    clustering_results.boxplot(
        column="last_event_SSY_elapsed_time_logratio", by="cluster_id", grid=False, color="grey", ax=ax[0]
    )
    clustering_results.boxplot(
        column="SSC_to_Q_peak_logratio", by="cluster_id", grid=False, color="grey", ax=ax[1]
    )
    plt.tight_layout()
    fig_list.append(fig)

    # seasonality and duration
    if with_year_column:
        fig, ax = plt.subplots(ncols=3)
        clustering_results.boxplot(
            column="seasonal_timing", by="cluster_id", grid=False, color="red", ax=ax[0]
        )
        clustering_results.boxplot(
            column="duration", by="cluster_id", grid=False, color="red", ax=ax[1]
        )
        clustering_results.boxplot(
            column="year", by="cluster_id", grid=False, color="red", ax=ax[2]
        )

    else:
        fig, ax = plt.subplots(ncols=2)
        clustering_results.boxplot(
            column="seasonal_timing", by="cluster_id", grid=False, color="red", ax=ax[0]
        )
        clustering_results.boxplot(
            column="duration", by="cluster_id", grid=False, color="red", ax=ax[1]
        )
    plt.tight_layout()
    fig_list.append(fig)

    return fig_list

def plot_cluster_number_GMM(n_components,aic,bic):
    fig,ax = plt.subplots()
    ax.plot(n_components, aic, label="AIC", color="red")
    ax.plot(n_components[aic.argmin()],aic.min(),"k.")
    ax.plot(n_components, bic, label="BIC", color="blue")
    ax.plot(n_components[bic.argmin()],bic.min(),"k.")
    ax.legend(loc="best")
    ax.set_xlabel("number of components")
    ax.set_xticks(n_components)
    return fig

def plot_cluster_scores(results):
    fig,ax = plt.subplots(nrows=2,sharex=True)
    ax1 = ax[0]
    ax2 = ax1.twinx()
    ax3 = ax[1]
    results.loc[
        :, ["silhouette_score_cos","silhouette_score_eucl"]
    ].plot(ax=ax1)
    results["calinski_harabasz_score"].plot(ax=ax2,color="green")
    ax1.set_ylabel("silhouette_score")
    ax2.set_ylabel("calinski_harabasz_score",color="green")
    ax1.set_xticks(results.index)
    results["davies_bouldin_score"].plot(ax=ax3,color="red",legend=True)
    ax3.set_xticks(results.index)
    return fig

def plot_dendrogram(model, **kwargs):
    """
    Adapted from: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(
        linkage_matrix,
        **kwargs,
    )
    plt.ylabel("Distances")

def plot_AGL_estimate_dist_threshold_from_quantile(dist_threshold,distances,q,ax=None):
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(distances, label="distances")
    ax.hlines(
        dist_threshold,
        xmin=0,
        xmax=len(distances),
        color="grey",
        label=f"{q} quantile",
    )
    ax.set_ylabel("Distances")
    ax.legend()

def plot_METS_clusters(data, labels, centeroids, alpha=0.1):
    fig, ax = plt.subplots(
        nrows=3,
        ncols=len(np.unique(labels)),
        figsize=(3 * len(np.unique(labels)), 7),
        sharey=True,
    )
    for c in np.unique(labels):
        data[labels == c]
        ax[0][c].set_title(f"Cluster {c+1}, n={(labels==c).sum()}")
        cluster_average = np.median(data[labels == c], axis=0)

        # plot streamflow
        ax[0][c].plot(data[labels == c, 0, :].T, alpha=alpha, color="blue")
        ax[0][c].plot(data[centeroids[c], 0, :].T, ":", color="darkblue")
        ax[0][c].plot(cluster_average[0], color="darkblue")

        # plot suspended sediment
        ax[1][c].plot(data[labels == c, 1, :].T, alpha=alpha, color="indianred")
        ax[1][c].plot(data[centeroids[c], 1, :].T, ":", color="darkred")
        ax[1][c].plot(cluster_average[1], color="darkred")

        # plot hysteresis
        ax[2][c].plot(
            data[labels == c, 0, :].T,
            data[labels == c, 1, :].T,
            color="grey",
            alpha=alpha,
        )
        ax[2][c].plot(
            data[centeroids[c], 0, :].T, data[centeroids[c], 1, :].T, ":", color="black"
        )
        ax[2][c].plot(cluster_average[0], cluster_average[1], color="black")

    ax[0][0].set_ylabel("Streamflow")
    ax[1][0].set_ylabel("Suspended sediment")
    ax[2][0].set_ylabel("Hysteresis")

def plot_single_cluster(data, labels, c, alpha=0.1, ax=None, ylabels = True):
    if ax is None:
        fig, ax = plt.subplots(
            nrows=3,
            figsize=(3, 7),
            sharey=True,
        )

    data[labels == c]
    ax[0].set_title(f"Cluster {c}, n={(labels==c).sum()}")
    cluster_average = np.median(data[labels == c], axis=0)

    # plot streamflow
    ax[0].plot(data[labels == c, 0, :].T, alpha=alpha, color="blue")
    ax[0].plot(cluster_average[0], color="darkblue")

    # plot suspended sediment
    ax[1].plot(data[labels == c, 1, :].T, alpha=alpha, color="indianred")
    ax[1].plot(cluster_average[1], color="darkred")

    # plot hysteresis
    ax[2].plot(
        data[labels == c, 0, :].T,
        data[labels == c, 1, :].T,
        color="grey",
        alpha=alpha,
    )
    ax[2].plot(cluster_average[0], cluster_average[1], color="black")
    
    if ylabels:
        ax[0].set_ylabel("Streamflow")
        ax[1].set_ylabel("Suspended sediment")
        ax[2].set_ylabel("Hysteresis")

def plot_cluster_zscore_averages(
    cluster_zscores_average,
    cluster_id,
    set_xlim_to_global=True,
    ax=None,
    color=[
        "red",
        "red",
        "brown",
        "brown",
        "brown",
        "blue",
        "blue",
        "blue",
        "grey",
        "grey",
        "green",
        "green",
        "green",
    ],
):
    if ax is None:
        fig, ax = plt.subplots()
    cluster_zscores_average.loc[cluster_id].plot.barh(
        ax=ax,
        color=color,
        width=0.8,
    )
    if set_xlim_to_global:
        ax.set_xlim(
            cluster_zscores_average.min().min() - 0.1,
            cluster_zscores_average.max().max() + 0.1,
        )
        
        
def plot_cluster_zscore_averages_with_error_range(
    cluster_zscores_average,
    cluster_zscores_lower,
    cluster_zscores_upper,
    cluster_id,
    set_xlim_to_global=True,
    error_color="black",
    ax=None,
    color=[
        "red",
        "red",
        "brown",
        "brown",
        "brown",
        "blue",
        "blue",
        "blue",
        "grey",
        "grey",
        "green",
        "green",
        "green",
    ],
):
    if ax is None:
        fig, ax = plt.subplots()
        
    error_range = pd.concat(
        [
            np.abs(cluster_zscores_lower.loc[cluster_id] - cluster_zscores_average.loc[cluster_id]),
            np.abs(cluster_zscores_upper.loc[cluster_id] - cluster_zscores_average.loc[cluster_id])
        ],
        axis=1,
    ).values.T
    cluster_zscores_average.loc[cluster_id].plot.barh(
        ax=ax,
        color=color,
        width=0.8,
        xerr=error_range,
        ecolor=error_color
    )
    if set_xlim_to_global:
        ax.set_xlim(
            cluster_zscores_average.min().min() - 0.1,
            cluster_zscores_average.max().max() + 0.1,
        )

        
def plot_cluster_result_with_zscore_averages(data,labels,cluster_zscores_average):
    fig, ax = plt.subplots(
        nrows=4,
        ncols=len(np.unique(labels)),
        figsize=(2.5 * len(np.unique(labels)), 7),
    )

    for c in np.unique(labels):

        plot_single_cluster(data, labels, c, alpha=0.1, ax=ax[:3, c])
        plot_cluster_zscore_averages(cluster_zscores_average,cluster_id=c,ax=ax[3][c])

        if c != 0:
            ax[0][c].set_ylabel("")
            ax[1][c].set_ylabel("")
            ax[2][c].set_ylabel("")
            ax[3][c].set_yticks([])

    for a in ax.ravel()[:-len(np.unique(labels))]:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()


def plot_cluster_zscores(event_metrics_zscores):
    ax = event_metrics_zscores.groupby("cluster_id").boxplot(
        grid=False,
        figsize=(10, 20),
        color="grey",
        medianprops=dict(linestyle="-", linewidth=2, color="k"),
        rot=90,
    )
    for a in ax:
        c = a.get_title()
        n = sum(event_metrics_zscores["cluster_id"] == int(float(c)))
        a.set_title(f"Cluster {c} (n={n})")
        a.hlines(
            y=0,
            xmin=0.5,
            xmax=a.get_xticks().max() + 0.5,
            zorder=1,
            color="grey",
            linewidth=1,
        )
    ax[0].set_ylabel("Z-Score")
    ax[2].set_ylabel("Z-Score")
    plt.tight_layout()


def plot_gmm(
    X, Y_, means, covariances, cluster_colors,  alpha=0, no_ticks=False, ax=None, xy=None, e=3
):
    x, y = xy or (0, 1)
    if ax is None:
        fig, ax = plt.subplots()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, cluster_colors)):
        ax.scatter(X[Y_ == i, x], X[Y_ == i, y], 1, color=color, label=f"Cluster{i}")

        if covariances.shape == means.shape[:1]:
            v = 2.0 * np.sqrt(2.0) * np.sqrt(covar)
            # Plot an ellipse to show the Gaussian component
            for a in np.linspace(1, e, 3):
                ell = mpl.patches.Ellipse(mean, v * a, v * a, color=color)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(alpha)
                ell.set_edgecolor(color)
                ell.set_facecolor("none")
                ax.add_artist(ell)
        else:
            v, w = scipy.linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[x] / scipy.linalg.norm(w[x])

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[y] / u[x])
            angle = 180.0 * angle / np.pi  # convert to degrees
            for a in np.linspace(1, e, 3):
                ell = mpl.patches.Ellipse(
                    mean, v[x] * a, v[y] * a, angle=180.0 + angle, color=color
                )
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(alpha)
                ell.set_edgecolor(color)
                ell.set_facecolor("none")
                ax.add_artist(ell)

    if no_ticks:
        plt.xticks(())
        plt.yticks(())


def violinplot_cluster_zscore(event_metrics_zscores,k,col_order,col_order_labels,col_colors,cluster_colors):
    fig, ax = plt.subplots(
        ncols=k,
        sharey=True,
        sharex=True,
        figsize=(len(col_order), int(len(col_order) / 3)),
    )
    for c, sub in event_metrics_zscores.groupby("cluster_id"):

        for j, m in enumerate(col_order):
            violins = ax[int(c)].violinplot(
                sub.loc[:, m].dropna(),
                positions=[j + 1],
                vert=False,
                showmedians=False,
                showmeans=True,
                widths=0.9,
            )
            violins["bodies"][0].set_edgecolor(col_colors[j])
            violins["bodies"][0].set_facecolor(col_colors[j])
            violins["bodies"][0].set_alpha(0.7)
            for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
                vp = violins[partname]
                vp.set_edgecolor(col_colors[j])
                vp.set_linewidth(2 if partname == "cmeans" else 1)

        ax[int(c)].vlines(0, 0, len(col_order) + 1, color="grey", alpha=0.5, linewidth=1)
        ax[int(c)].set_xlabel("Z-Score")
        ax[int(c)].set_title(f"Cluster{int(c)}", color=cluster_colors[int(c)], fontsize=16)


    ax[0].set_ylim(0, len(col_order) + 1)
    ax[0].set_xlim(-7, 7)
    plt.yticks(range(1, len(col_order) + 1), col_order_labels)
    
    return ax