
import copy as cp
import numpy as np

from numpy import typing as npt
from typing import Any, Callable

from ..metrics import euclidean


class DBScan():
    '''
    Class to perform clustering through the DBScan algorithm. This separates samples into clusters
    based on their density. The algorithm generates the clusters organically from the data, no need
    to specify the number of clusters. The DBScan algorithm is a density-based clustering algorithm
    and uses the notions of core-points, border-points and noise.

    Parameters:
    ===============================================================================================

        epsilon (float, no default): The radius, i.e. size of the ball, around a sample point in
        which to search for neighbours.

        minimum_neighbour_samples (int, default = 5): Minimum number of neighbours a sample point
        needs to have within its epsilon-neighbourhood for it to be considered a core point.

        distance_metric (Callable, default = euclidean): The distance metric to use for the cluster
        centroid computation, i.e. how to determine which cluster is closest to a specific point.
        This can be any user-defined function with its call-signature taking to numpy ndarrrays
        initially and possible keyword arguments which are specified in the
        distance_metric_kwargs-dictionary. Some popular distance metrics are specified in the
        distances-library like Euclidean, Manhattan, and Chebyshev.
    '''


    def __init__(self, epsilon: float, minimum_neighbour_samples: int = 5, 
                 distance_metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = euclidean,
                 distance_metric_kwargs: dict[str, Any] = {}) -> None:

        self.epsilon: float                 = epsilon
        self.minimum_neighbour_samples: int = minimum_neighbour_samples

        self.metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = distance_metric
        self.metric_kwargs: dict[str, Any]                             = distance_metric_kwargs


    def _get_neighbourhood(self, sample_index: int, distances: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        '''
        Get the neighbourhood of a sample point, i.e. all indices within a certain distance,
        specified by epsilon on initialization, of the sample point. Uses a precomputed set of
        distances for better performance.
        '''

        neighbours = np.where(distances[sample_index] < self.epsilon)[0]
        return np.delete(neighbours, np.where(neighbours == sample_index))
    

    def is_core(self, sample_index: int, data: npt.NDArray[np.float_]) -> bool:
        '''
        Given a sample index and an array of datapoints, check whether or not the given
        sample point is a core point in the distribution or not.
        '''

        distances = self.metric(data, data, **self.metric_kwargs)
        return len(self._get_neighbourhood(sample_index = sample_index, distances = distances)) > self.minimum_neighbour_samples
    

    def find_core_points(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        '''
        Given the dataset, find all core points of the distribution. This method is a convenience
        function and not used internally. It can give quick insight into the distribution of data
        for a simple analysis.
        '''

        distances = self.metric(data, data, **self.metric_kwargs)
        return np.where((np.sum(distances < self.epsilon, axis = 1) - 1) >= self.minimum_neighbour_samples)[0]
    

    def _expand_cluster(self, sample_index: int, neighbours: npt.NDArray[np.int_], cluster_label: int):
        '''
        Given a sample point, i.e. index of a sample point, and its neighbourhood a cluster is
        created by iteratively expanding the cluster from the core points until the border is
        reached. A border-point is a point that lies within the neighbourhood of a core-point but
        does not have enough neighbours by itself to become a cluster-seed.
        '''

        self.labels[sample_index] = cluster_label

        for neighbour in neighbours:

            ## Check if the sample point has been considered previously. If this is the case and it
            ## was designated as noise, i.e. label equal to -1, it becomes part of the cluster and
            ## no checking of its neighbourhood to expand the cluster is needed.
            if self.labels[neighbour] == -1:
                self.labels[neighbour] = cluster_label

            ## If the sample has not been considered yet, assign it to the cluster and check its
            ## neighbourhood.
            elif self.labels[neighbour] == 0:
                self.labels[neighbour] = cluster_label

                neighbourhood_of_neighbour = self._get_neighbourhood(sample_index = neighbour, distances = self.distances)

                ## If the neighbourhood is large enough, call the expand_cluster method on the new
                ## core-point to keep expanding the cluster until its borders are reached.
                if len(neighbourhood_of_neighbour) >= self.minimum_neighbour_samples:
                    self._expand_cluster(sample_index = neighbour, neighbours = neighbourhood_of_neighbour, cluster_label = cluster_label)


    def fit(self, X: npt.NDArray[np.float_], y = None) -> None:
        '''
        This method performs the actual clustering following the DBScan algorithm. It only requires
        the input data and the, earlier specified, epsilon and minimum_sample_points parameters
        which are stored internally.
        '''

        self.data: npt.NDArray[np.float_] = cp.deepcopy(X)
        self.labels: npt.NDArray[np.int_] = np.zeros(shape = (self.data.shape[0], ), dtype = np.int_)

        cluster_label: int = 1

        ## Precompute the distances all at once, so they do not need to be computed at each
        ## iteration again. This is a big performance benefit typically.
        self.distances = self.metric(self.data, self.data, **self.metric_kwargs)

        for sample_index, sample_point in enumerate(self.data):

            ## Check if the label of the sample point is set to zero, meaning it has not been
            ## visited yet. There is no need to visit a given sample twice because it might
            ## get assigned to a cluster based on previous sample points.
            if self.labels[sample_index] == 0:

                neighbours = self._get_neighbourhood(sample_index = sample_index, distances = self.distances)

                ## If the epsilon-neighbourhood is not large enough, simply mark the sample point
                ## as noise by setting its label to -1.
                if len(neighbours) < self.minimum_neighbour_samples:
                    self.labels[sample_index] = -1

                ## Otherwise search the neighbourhood and keep expanding it until the border-points
                ## are reached. After this is done increase the running cluster label so no two
                ## clusters get the same label.
                else:
                    self._expand_cluster(sample_index, neighbours = neighbours, cluster_label = cluster_label)
                    cluster_label += 1


    def fit_predict(self, X: npt.NDArray[np.float_], y = None) -> npt.NDArray[np.int_]:
        '''
        Convenience method to perform the clustering and return the labels as well.
        '''

        self.fit(X = X)
        return self.labels


