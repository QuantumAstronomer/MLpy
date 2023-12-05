
import copy as cp
import numpy as np

from numpy import typing as npt
from typing import Any, Callable, Literal

from ..metrics import euclidean


class KMeans():
    '''
    Class to perform clustering through the K-means algorithm by separating samples in a pre-
    specified number of clusters of equal variance minimizing the inertia or within-cluster
    sum-of-squares (WCSS). The algorithm requires the number of clusters to be specified and
    scales well to large numbers of samples. However, it does not perform as well for clusters
    of unequal variance, unequal size, or anisotropically distributed clusters.

    Parameters:
    ===============================================================================================

        cluster_count (int, no default): Number of clusters to form, i.e. the number of centroids
        to generate around which the data is clustered.

        initializtion_method ({'k-means++', 'random', 'random-data'}, default = 'k-means++'): 
        Method used for generating the initial clusters:
            - k-means++ selects initial cluster centroids using an empirical probability
              distribution of a points contribution to the overall inertia to sample the initial
              clusters, typically speeding up convergence and leading to more robust centroids,
              i.e. centroids do not differ from one run to the next.
            - random picks the initial cluster centroids from a uniform distribution within the
              bounds of the data, i.e. the extreme values are used as the bounds in each feature
              dimension.
            - random-data selects initial clusters from the data sample.

        initialization_count ('auto' or int, default = 'auto'): Number of times the k-means is run
        with different centroid seeds. Final result is the best output of the runs, i.e. the one
        with the lowest inertia. When set to 'auto', an appropriate number of trials is chosen
        depending on the initialization method (1 if k-means++, 10 runs otherwise).

        max_iteration (int, default = 20): Maximum number of iteration of the k-means algorithm in
        a single run.

        tolerance (float, default = 1e-4): Relative tolerance regarding the L2-norm of the
        difference in cluster centers of two iterations. It is used to declare convergence and
        early halting of the clustering when cluster centroids are no longer moving significantly.
        This speeds up the clustering method, reducing computation time significantly.

        algorithm ({'lloyd', 'macqueen', 'hartigan-wong'}, default = 'lloyd'): K-means algorithm
        to use to iteratively update the cluster centroids.

        distance_metric (Callable, default = euclidean): The distance metric to use for the cluster
        centroid computation, i.e. how to determine which cluster is closest to a specific point.
        This can be any user-defined function with its call-signature taking to numpy ndarrrays
        initially and possible keyword arguments which are specified in the
        distance_metric_kwargs-dictionary. Some popular distance metrics are specified in the
        distances-library like Euclidean, Manhattan, and Chebyshev.
    '''

    def __init__(self, cluster_count: int,
                 initialization_method: Literal['k-means++', 'random', 'random-data'] = 'k-means++',
                 initialization_count: Literal['auto'] | int = 'auto',
                 max_iteration: int = 20,
                 tolerance: float = 1e-4,
                 algorithm: Literal['lloyd', 'macqueen', 'hartigan-wong'] = 'lloyd',
                 distance_metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = euclidean,
                 distance_metric_kwargs: dict[str, Any] = {}) -> None:
        
        self.cluster_count: int = cluster_count
        self.max_iteration: int = max_iteration
        self.tolerance: float   = tolerance

        self.metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = distance_metric
        self.metric_kwargs: dict[str, Any]                             = distance_metric_kwargs

        if initialization_method not in ['k-means++', 'random', 'random-data']:
            raise ValueError(f'Provided initialization method ({initialization_method}) not supported.')
        
        self.initialization_method: str = initialization_method
        
        if algorithm not in ['lloyd', 'macqueen', 'hartigan-wong']:
            raise ValueError(f'Provided algorithm ({algorithm}) not supported.')
        
        self.algorithm: str = algorithm

        if not isinstance(initialization_count, int) and not initialization_count == 'auto':
            raise ValueError(f'Initialization count should be an integer or "auto", received {initialization_count} of type {type(initialization_count)}.')
        
        if initialization_count == 'auto':
            self.initialization_count = 10 if self.initialization_method in ['random', 'random-data'] and self.algorithm != 'hartigan-wong' else 1
        else:
            self.initialization_count = initialization_count


    def _initialize_centroids(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Wrapper method to initialize the cluster centroids easily based on the data using the
        desired initialization method. A check is performed that the number of clusters does not
        exceed the number of datapoints, as this would result in a degenerate problem.
        '''

        if data.shape[0] < self.cluster_count:
            raise ValueError(f'The number of samples ({data.shape[0]}) can not be less than the number of clusters ({self.cluster_count}) to find.')

        if self.initialization_method == 'k-means++':
            return self._kmeanspp(data = data)
        
        elif self.initialization_method == 'random':
            return self._random(data = data)
        
        elif self.initialization_method == 'random-data':
            return self._random_data(data = data)
        
        else:
            raise ValueError(f'Initialization method {self.initialization_method} is not supported.')
        

    def _kmeanspp(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Employ the k-means++ algorithm to initialize cluster centroids, this method is preferred,
        as it leads to the most robust cluster centers and the most efficient convergence.
        '''

        initial_centroids = data[np.random.choice(data.shape[0], size = 1)]

        for i in range(1, self.cluster_count):

            ## Compute distance (squared) to the nearest already picked centroid for each datapoint
            ## No need to mask datapoints that are centroids as their probability will be set to
            ## zero as their minimum distance is zero.
            distances = np.nanmin(self.metric(data, initial_centroids, **self.metric_kwargs), axis = 1)**2

            ## Turn the distances in to proper probabilities by making them sum to one.
            probabilities = distances / np.nansum(distances)

            ## And pick the new data point at random and add it to the list of centroids.
            next_centroid = data[np.random.choice(data.shape[0], size = 1, p = probabilities)]
            initial_centroids = np.vstack([initial_centroids, next_centroid])

        return initial_centroids


    def _random(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Initialize the cluster centroids at random within the range of the data.
        '''
        return np.random.uniform(low = np.min(data, axis = 0), high = np.max(data, axis = 0), size = (self.cluster_count, data.shape[1]))


    def _random_data(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Pick cluster centroids from the input-data to cluster at random.
        '''
        return data[np.random.choice(a = data.shape[0], size = self.cluster_count, replace = False)]
    

    def _get_inertia(self, data: npt.NDArray[np.float_], centroids: npt.NDArray[np.float_]) -> np.float_:
        '''
        The inertia is a measure of the quality of clustering, it is also known as the within-
        cluster sum-of-squares. A lower inertia is seen as a more optimized result.
        '''
        return np.sum(np.min(self.metric(data, centroids, **self.metric_kwargs), axis = -1))
    

    def _get_inertia_multiple(self, data: npt.NDArray[np.float_], centroids: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        This method calculates the inertia for multiple sets of centroids, mainly used by the
        Hartigan-Wong algorithm of clustering.
        '''

        if centroids.ndim != 3:
            raise ValueError(f'Centroids-array of shape {centroids.shape} with dimension {centroids.ndim} '
                             f'can not be used to calculate multiple inertias.')
        return np.array([self._get_inertia(data = data, centroids = centroid) for centroid in centroids])
    

    def _lloyd(self, data: npt.NDArray[np.float_], initial_centroids: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], int]:
        '''
        Perform K-means clustering through the lloyd algorithm. This is the default algorithm as
        it is most intuitively understandable and computationally simple.
        '''

        termination_iteration: int = 0
        centroids = initial_centroids
        previous_centroids = cp.deepcopy(centroids)

        for iteration in range(self.max_iteration):

            labels = np.argmin(self.metric(data, centroids, **self.metric_kwargs), axis = -1)

            for k in range(self.cluster_count):
                centroids[k] = np.nanmean(data[labels == k], axis = 0)

            if np.all(np.linalg.norm((centroids - previous_centroids) / previous_centroids, axis = 1) < self.tolerance):
                break

            termination_iteration = iteration + 1
            previous_centroids = cp.deepcopy(centroids)

        return centroids, termination_iteration
    

    def _macqueen(self, data: npt.NDArray[np.float_], initial_centroids: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], int]:
        '''
        The macqueen algorithm for clustering is more sophisticated and computationally expensive
        compared to the lloyd algorithm, as it loops through the datapoints 1-by-1. However, it
        has the potential to converge in less iterations typically.
        '''
        
        termination_iteration: int = 0
        centroids = initial_centroids
        previous_centroids = cp.deepcopy(centroids)
        labels = -1 * np.ones(shape = data.shape[0])

        for iteration in range(self.max_iteration):

            for i, datum in enumerate(data):
                distance = self.metric(np.expand_dims(datum, axis = 0), centroids)
                label = np.argmin(distance)
                labels[i] = label
                centroids[label] = np.nanmean(data[labels == label], axis = 0)

            if np.all(np.linalg.norm((centroids - previous_centroids) / previous_centroids, axis = 1) < self.tolerance):
                break

            termination_iteration = iteration + 1
            previous_centroids = cp.deepcopy(centroids)

        return centroids, termination_iteration


    def _hartigan_wong(self, data: npt.NDArray[np.float_], initial_centroids: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], int]:
        '''
        In the Hartigan-Wong algorithm, datapoints are assigned to cluster centroids. However,
        this is not done like the Lloyd or MacQueen algorithms by minimizing the distance to
        the cluster centers. In Hartigan-Wong, the WCSS is optimized to find the optimal cluster
        centers. This is a global measure of the goodness-of-fit of the clusters. It is therefore
        less likely to get stuck in local optima at the cost of great computational expense.
        '''
        
        termination_iteration: int = 0
        centroids = initial_centroids
        previous_centroids = cp.deepcopy(centroids)
        labels = np.argmin(self.metric(data, centroids, **self.metric_kwargs), axis = -1)

        for iteration in range(self.max_iteration):

            for i, (datum, old_label) in enumerate(zip(data, labels)):
                labels[i] = -1
                centroids[old_label] = np.nanmean(data[labels == old_label], axis = 0)
                trial_centroids = np.stack(self.cluster_count * [centroids], axis = 0)

                for k in range(self.cluster_count):
                    labels[i] = k
                    trial_centroids[k, k] = np.nanmean(data[labels == k], axis = 0)

                labels[i] = np.argmin(self._get_inertia_multiple(data = data, centroids = trial_centroids))

                for k in range(self.cluster_count):
                    centroids[k] = np.nanmean(data[labels == k], axis = 0)
            
            if np.all(np.linalg.norm((centroids - previous_centroids) / previous_centroids, axis = 1) < self.tolerance):
                break

            termination_iteration = iteration + 1
            previous_centroids = cp.deepcopy(centroids)

        return centroids, termination_iteration
        


    def _single_fit(self, data: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], int]:
        '''
        This method performs a singular fitting procedure to the data given a specific
        initialization. It can then be called multiple times according to the number of
        initializations that will be performed.
        '''

        initial_centroids = self._initialize_centroids(data = data)

        if self.algorithm == 'lloyd':
            return self._lloyd(data = data, initial_centroids = initial_centroids)
        
        elif self.algorithm == 'macqueen':
            return self._macqueen(data = data, initial_centroids = initial_centroids)
        
        elif self.algorithm == 'hartigan-wong':
            return self._hartigan_wong(data = data, initial_centroids = initial_centroids)
        
        else:
            raise ValueError(f'Provided algorithm ({self.algorithm}) not supported. Choose from [lloyd, macqueen, or hartigan-wong]')
    

    def fit(self, X: npt.NDArray[np.float_], y = None) -> None:
        '''
        Performs multiple fitting procedures, according to the number of initializations that are
        to be performed. It keeps track of the best fitting one according to the inertia it has for
        which it records the inertia, number of iterations it took to reach convergence and the 
        corresponding centroids of course.
        '''

        best_centroids = np.empty(shape = (self.cluster_count, X.shape[1]))
        best_inertia = np.inf
        best_iterations = np.inf

        for _ in range(self.initialization_count):
            centroids, iterations = self._single_fit(data = X)
            inertia = self._get_inertia(data = X, centroids = centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_iterations = iterations

        self.centroids = best_centroids
        self.inertia = best_inertia
        self.convergence_iterations = best_iterations
        self.labels = np.argmin(a = self.metric(X, self.centroids, **self.metric_kwargs), axis = 1)


    def fit_predict(self, X: npt.NDArray[np.float_], y = None) -> npt.NDArray[np.int_]:
        '''
        Convenience method to perform the clustering and return the labels as well.
        '''

        self.fit(X = X)
        return self.labels