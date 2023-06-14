"""
Basis implementation of Kmedoids algorithm for multi-variate time series that uses dynamic time warping-dependent for distance measure. 

The following code has beed taken from the following repository:
https://github.com/ali-javed/Multivariate-Kmedoids.git

The code has been restructured, but is in essential unaltered from the original (Version 1.0).

If you use this code in your work, please cite it as:
@article{Javed_CoRR2019,
  author    = {Ali Javed and Scott Hamshaw and Donna M. Rizzo and Byung Suk Lee},
  title     = {Hydrological and Suspended Sediment Event Analysis using Multivariate Time Series Clustering},
  journal   = {CoRR},
  year      = {2019},
  archivePrefix = {arXiv},
  bibsource = {dblp computer science bibliography, https://dblp.org}

Peer-reviewed paper:
https://doi.org/10.1016/j.jhydrol.2020.125802

Version 1.0:
author: Ali Javed (AJ)
email: ajaved@uvm.edu

From Version 1.1:
author: Amalie Skålevåg (AS)
email: skalevag2@uni-potsdam.de

Version history:
################
Version 1.0 : basis implementation of Kmedoids algorithm for multi-variate time series that uses dynamic time warping -dependent for distance measure (AJ)
Version 1.1 : moved DTW and Kmedoids clustering to same script, added docstrings, and better error handling (AS)
Version 1.2 : added logging (AS)



########################################################################
    
    #Inputs
    #timeseries : shape x by y by z, where x is the number of time series to cluster, y is the dimensionality, and z is the length of each timeseries.
    #k          : is the number of clusters.
    #max_iter   : maximum iterations to perform incase of no convergence.
    #window_size: is the dynamic time wrapping window size as a ration i.e. 0 to 1.
    
    
    #outputs
    #labels    : cluster number for each time series.
    #sse_all   : sum of squared errors in each iteration. 
    #j         : number of iterations performed.
    #closest_observations_prev: Centeroids
    
########################################################################

This is code for multivariate time series dynamic time warping using euclidean distance. The code by default calculated dynamic time warping dependent. If you are interested in dynamic time warping independent, simply call the dtw_d function on each variable separately and sum the resulting distances.
"""
#If you use the code in your work please cite as
#@misc{DTW_D,
#title={Multivariate time series dynamic time warping using euclidean distance},
#author={Ali Javed},
#year={2019},
#month={November},
#note = {\url{https://github.com/ali-javed/dynamic-time-warping}}
#}

import numpy as np
import pandas as pd
import random
import logging
from watersedimentpulses.utils.tools import log
import watersedimentpulses.utils.visualise
import multiprocessing as mp
from itertools import repeat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sq_euc(s1, s2):
    """Squared euclidean distance
    
    author: Ali Javed 
    email: ajaved@uvm.edu
    Version history:
    Version 1 . basis implementation of dynaimc time warping dependant. 
    Version 2 (7 Nov 2019). changed variable names to be more representative and added comments.

    Args:
        s1 (array-like): signal 1, size 1 * m * n. where m is the number of variables, n is the timesteps.
        s2 (array-like): signal 2, size 1 * m * n. where m is the number of variables, n is the timesteps.


    Returns:
        float: Squared euclidean distance
    """
    dist = ((s1 - s2) ** 2)
    return dist.flatten().sum()


def dtw_d(s1: np.ndarray, s2: np.ndarray, w: float):
    """A variant of dynamic time warping (DTW) to calculate the distance between two multivariate times series.

    author: Ali Javed 
    email: ajaved@uvm.edu
    Version 1 . basis implementation of dynaimc time warping dependant. 
    Version 2 (7 Nov 2019). changed variable names to be more representative and added comments.

    Args:
        s1 (np.ndarray): signal 1, size 1 * m * n. where m is the number of variables, n is the timesteps.
        s2 (np.ndarray): signal 2, size 1 * m * n. where m is the number of variables, n is the timesteps.
        w (float): window parameter, percent of size and is between0 and 1. 0 is euclidean distance while 1 is maximum window size.

    Returns:
        float: DTW distance
    """

    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    s1_shape = np.shape(s1)
    s2_shape = np.shape(s2)
    
    if w<0 or w>1:
        raise ValueError("w should be between 0 and 1")
    if s1_shape[0] >1 or s2_shape[0] >1:
        raise ValueError("Please check input dimensions.")
    if s1_shape[1] != s2_shape[1]:
        raise ValueError("Please check input dimensions. Number of variables not consistent.")
    if s1_shape[2] != s2_shape[2]:
        raise ValueError("Warning: Length of time series not equal")
        
    #if window size is zero, it is plain euclidean distance
    if w ==0:
        dist = np.sqrt(sq_euc(s1, s2))
        return dist


    #get absolute window size
    w = int(np.ceil(w * s1_shape[2]))

    #adapt window size
    w=int(max(w, abs(s1_shape[2]- s2_shape[2])));
        
        
    #initilize    
    DTW = {}
    for i in range(-1, s1_shape[2]):
        for j in range(-1, s2_shape[2]):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0


    for i in range(s1_shape[2]):
        for j in range(max(0, i - w), min(s2_shape[2], i + w)):
            #squared euc distance
            dist = sq_euc(s1[0,:,i], s2[0,:,j])
            #find optimal path
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    dist = np.sqrt(DTW[s1_shape[2] - 1, s2_shape[2] - 1])
    
    
    return dist


#if you use this code in your work, please cite it as
#@article{Javed_CoRR2019,
#  author    = {Ali Javed and Scott Hamshaw and Donna M. Rizzo and Byung Suk Lee},
#  title     = {Hydrological and Suspended Sediment Event Analysis using Multivariate Time Series Clustering},
#  journal   = {CoRR},
#  year      = {2019},
#  archivePrefix = {arXiv},
#  bibsource = {dblp computer science bibliography, https://dblp.org}
#  }    
 

def assign(timeseries: np.ndarray, k: int, centeroids: np.ndarray, window_size: float):
    """
    #inputs: 
    #timeseries : shape x by y by z, where x is the number of time series to cluster, y is the dimensionality, and z is the length of each timeseries.
    #k          : Number of clusters
    #centeroids : Index of each medoid in timeseries
    #window size: Absolute window size i.e. number of time steps to consider in window
    
    #output:
    #labels   : Assigned labels to each time series in the input of all time series
    #sse      : Sum of squared error
    
    """
    
    labels = []
    sse = 0
    
    #for all points in time series
    for i in range(0, len(timeseries)):
        #initilize distance to each centeroid as infinity
        dist_to_center = np.zeros(k)
        dist_to_center = dist_to_center+float('inf')

        #calculate distance to each centeroid
        for j in range(0, len(centeroids)):
            ob1 = timeseries[i]
            ob2 = timeseries[centeroids[j]]
            ob1 = np.reshape(ob1, (1, np.shape(ob1)[0], np.shape(ob1)[1]))
            ob2 = np.reshape(ob2, (1, np.shape(ob2)[0], np.shape(ob2)[1]))
            dist_to_center[j] = dtw_d(ob1,ob2,window_size)
            
        #find closest medoid to ith time series

        c = np.argmin(dist_to_center)
        labels.append(c)
        

        #error checking
        if dist_to_center[c]!=dist_to_center[c]:
            raise ValueError('Possible nan values in data')
            
        sse = sse + dist_to_center[c]

    labels = np.asarray(labels)
    if len(set(labels))< k:
        #one cluster never got made
        # create random centeroid for that cluster
        for i in range(k):
            if i not in set(labels):

                centeroid_index = random.sample(range(0, len(timeseries)), 1)
                labels[centeroid_index]=i

    return labels, sse



def find_central_node(timeseries: np.ndarray,window_size: float,labels,k: int):
    labels = np.asarray(labels)
    centeroids = []
    closest_observations = []
    
    #for each cluster
    for i in range(k):
        
        #get observations of assigned to that cluster
        elements = np.where(labels == i)[0]
        
        #we will make each observation as a centeroid. Set the SSE to zero if a particular observation is the centeroid
        distances = np.zeros(len(elements))
        
        #for each observation in the cluster
        for j in range(0,len(elements)):

            #make observation temperarily as centeroid
            temp_centeroid = timeseries[elements[j]]
            
            #for each observation in the cluster measure its distance to temperary centeroid
            for l in range(0,len(elements)):
                ob1 = temp_centeroid
                ob2 = timeseries[elements[l]]
                ob1 = np.reshape(ob1, (1, np.shape(ob1)[0], np.shape(ob1)[1]))
                ob2 = np.reshape(ob2, (1, np.shape(ob2)[0], np.shape(ob2)[1]))
                distances[j] = distances[j]+dtw_d(ob1,ob2,window_size)
                
                
        #select observation that minimizes SSE

        c = np.argmin(distances)
        #the actual time series of centeroid
        centeroids.append(timeseries[elements[c]])
        #the index number of centeroid in time series
        closest_observations.append(elements[c])
        
        
    if len(centeroids)<k:
        raise ValueError('number of centeroids are not equal to k')
        
    
    return np.asarray(centeroids), np.asarray(closest_observations)
        
    

@log(logger)
def MultivariateKmedoids(timeseries, k, max_iter, window_size):
    """
    Author: Ali Javed
    Date September 17th 2019. 
    Email: ali.javed@uvm.edu
    Please note the code is not maintained. 
    
    Inputs
    timeseries : shape x by y by z, where x is the number of time series to cluster, y is the dimensionality, and z is the length of each timeseries.
    k          : is the number of clusters.
    max_iter   : maximum iterations to perform incase of no convergence.
    window_size: is the dynamic time wrapping window size as a ration i.e. 0 to 1.
    
    
    outputs
    labels    : cluster number for each time series.
    sse_all   : sum of squared errors in each iteration. 
    j         : number of iterations performed.
    closest_observations_prev: Centeroids
    """

    #create a empty labels array set to -1
    labels = np.ones(len(timeseries))
    labels = labels * -1

    #number of series
    dimensions = np.shape(timeseries)[1]
    
    # DECLARE LISTS
    sse_all = [] # this list is not used, could modify code to store SSE of all runs here


    #create random centeroids
    centeroid_index = random.sample(range(0, len(timeseries)), k)


    #take centeroids of all series
    centeroids = np.zeros((k, dimensions, np.shape(timeseries)[2]))
    for i in range(0,dimensions):
        centeroids_temp = timeseries[:,i,:][centeroid_index]
        centeroids[:,i,:] = centeroids_temp
        #centeroids.append(centeroids_temp)
        
    #reshape centeroids for the first index to represent centeroid
    #centeroids = np.asarray(centeroids)
    #centeroids = np.reshape(centeroids,(k,dimensions,np.shape(timeseries)[2]))
    
    sse_current = float('inf')
    conv_itr = 0
    closest_observations = centeroid_index
    
    #start iterations
    for j in range(0, max_iter):
        sse_previous = sse_current
        closest_observations_prev = closest_observations
        
        #assign observation to each centeroid
        labels, sse_current = assign(timeseries, k, closest_observations, window_size)
        
        if sse_current != sse_current:
            print('Error 888: value of k voilated or presence of nan values.')
            break

        # calculate new centeroids
        centeroids, closest_observations= find_central_node(timeseries,window_size,labels,k)
        
        # TERMINATION CRITERIA
        if np.abs(sse_current - sse_previous) == 0:
            conv_itr += 1
            if conv_itr > 1:
                labels, sse_current = assign(timeseries, k, closest_observations, window_size)
                break
        
    return labels, sse_current, j, closest_observations_prev


def run_MultivariateKmedoids_cluster_single(data, event_ids, k, max_iter, window_size, ODIR):
    logger.info(f"Starting METS clustering with {k} clusters...")
    labels, sse_current, iterations, centeroids = MultivariateKmedoids(
        timeseries=data, k=k, max_iter=max_iter, window_size=window_size
    )

    # save results of clustering with parameters
    events_per_cluster = np.array([(labels == c).sum() for c in np.unique(labels)])
    results = [
        (
            k,
            max_iter,
            window_size,
            sse_current,
            iterations,
            round(np.median(events_per_cluster)),
            events_per_cluster.min(),
            events_per_cluster.max(),
            ",".join([str(c) for c in centeroids]),
        )
    ]
    logger.info(f"Finished METS clustering with {k} clusters.")


    results = pd.DataFrame(
        results,
        columns=[
            "n_clusters",
            "max_iterations",
            "window_size",
            "SSE",
            "n_iterations",
            "n_events_per_cluster_median",
            "n_events_per_cluster_min",
            "n_events_per_cluster_max",
            "centeroids",
        ],
    )
    k_labels = pd.DataFrame(np.array(labels).T, columns=[f"{k}_clusters"], index=event_ids)
    k_labels.index.name = "event_id"
    k_labels = k_labels.reset_index()
    
    # save to file
    k_labels.to_csv(ODIR.joinpath(f"tmp_{k}_labels.csv"),index=False)
    results.to_csv(ODIR.joinpath(f"tmp_{k}_results_eval.csv"),index=False)


@log(logger)
def run_MultivariateKmedoids_cluster_range(data, k_range, max_iter, window_size, event_ids, ODIR, max_cores = 5):
    """
    Author: Amalie Skålevåg 
    Email: skalevag2@uni-potsdam.de
    """
    locallog = logging.FileHandler(filename=ODIR.joinpath(f'{__name__.split(".")[-1]}.log'),mode='w')
    locallog.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    locallog.setFormatter(formatter)
    logger.addHandler(locallog)

    logger.info(f"METS clustering with cluster range from {k_range[0]} to {k_range[-1]}, max_iter = {max_iter}")
    logger.info(f"{data.shape[0]} events")

    results = []
    k_labels = []

    # parallel processing
    cores = min(max_cores,mp.cpu_count()-1) # max 5 cores or one less than system
    logger.info(f"Parallelising with {cores} cores.")
    # apply parallel processing
    with mp.Pool(cores) as pool:
        pool.starmap(run_MultivariateKmedoids_cluster_single, zip(repeat(data),repeat(event_ids),k_range,repeat(max_iter),repeat(window_size),repeat(ODIR)))

    # import results from temporary files
    k_labels = pd.concat([pd.read_csv(file,index_col=0) for file in ODIR.glob("tmp_*_labels.csv")],axis=1)
    results = pd.concat([pd.read_csv(file) for file in ODIR.glob("tmp_*_results_eval.csv")])
    results = results.sort_values("n_clusters").reset_index(drop=True) # sort results according to cluster number
    # remove temporary files
    [file.unlink() for file in ODIR.glob("tmp_*_results_eval.csv")]
    [file.unlink() for file in ODIR.glob("tmp_*_labels.csv")]
    
    # save to file
    k_labels.to_csv(ODIR.joinpath("METS_clustering_results_labels.csv"),index=True)
    results.to_csv(ODIR.joinpath("METS_clustering_results_eval.csv"),index=False)
    # plot clustering results
    watersedimentpulses.utils.visualise.plot_MultivariateKmedoids_eval(results,ODIR=ODIR)
    
    return k_labels, results
