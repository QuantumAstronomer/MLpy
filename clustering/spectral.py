import copy as cp
import numpy as np

from numpy import typing as npt
from scipy.linalg import qr, svd
from scipy.sparse.linalg import eigs
from typing import Any, Callable, Literal

from ..metrics import euclidean
from ..clustering import KMeans


class Spectral():
    '''
    Class to perform clustering following the spectral clustering algorithm. Spectral clustering
    relies on constructing a graph Laplacian from the connectivity matrix and performs a non-linear
    transformation through the eigenvector calculation of this graph Laplacian. This projects the
    feature space into a lower dimensional vector space where clustering is easier to perform.
    Spectral clustering performs well when no suitable measure to capture the cluster "identity"
    exists, e.g. when clusters are concentric circles or half-moons.

    Parameters:
    ===============================================================================================

        cluster_count (int, no default): The number of clusters that are to be formed.

        eigen_vectors (int | None, default = None): The number of eigenvectors to compute and use
        in the actual clustering procedure. If None, the number of eigenvectors is set to the
        number of clusters to form. Note that the QR-clustering method always forms a number of
        clusters equal to the number of eigenvectors. The best value of eigen vectors is often
        equal to the number of clusters to form.

        neighbourhood_method ({'knn', 'epsilon'} default = 'knn'): Which method to use for to find/
        compute the neighbourhoods of the data points and thus to find the connectivity matrix.
        'knn' uses a k-nearest neighbours method and thus results in a non-symmetric connectivity
        matrix. 'epsilon' uses a ball centered on the data point with a fixed radius to find all
        points it is connected to.

        neighbourhood_count (int, default = 5): Number of nearest neighbours to find when the
        neighbourhood_method is set to 'knn'.

        epsilon (float, default = .5): Radius of the ball when the neighbourhood_method is set to
        the 'epsilon' method, i.e. neighbours are to be found within a radius epsilon of the data
        point in question.

        distance_metric (Callable, default = euclidean): The distance metric to use for the cluster
        centroid computation, i.e. how to determine which cluster is closest to a specific point.
        This can be any user-defined function with its call-signature taking to numpy ndarrrays
        initially and possible keyword arguments which are specified in the
        distance_metric_kwargs-dictionary. Some popular distance metrics are specified in the
        distances-library like Euclidean, Manhattan, and Chebyshev.

        clustering_method ({'k-means', 'qr-clustering'}, default = 'k-means'): Specifies which
        method is to be used to assgin the cluster labels after the graph Laplacian has been
        embedded into eigenvectors. 'k-means' uses the well-known k-means algorithm while 
        'qr-clustering' uses the recent algorithm developed by Damle et al. 2019,
        (https://doi.org/10.1093/imaiai/iay008). In a future version this will be extended to
        include the discretized clustering technique proposed by Yu and Shi 2003, 
        (https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf).
    '''

    def __init__(self, cluster_count: int,
                 eigen_vectors: int | None = None,
                 neighbourhood_method: Literal['knn', 'epsilon'] = 'knn',
                 neighbour_count: int = 5,
                 epsilon: float = .5,
                 distance_metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = euclidean,
                 distance_metric_kwargs: dict[str, Any] = {},
                 clustering_method: Literal['k-means', 'qr-clustering'] = 'k-means') -> None:
        
        self.cluster_count: int = cluster_count
        self.neighbour_count: int = neighbour_count
        self.epsilon: float = epsilon
        self.metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = distance_metric
        self.metric_kwargs: dict[str, Any] = distance_metric_kwargs

        if neighbourhood_method not in ['knn', 'epsilon']:
            raise ValueError(f'Neighbourhood method {neighbourhood_method} not supported, choose from (knn | epsilon)...')
        self.neighbourhood_method: str = neighbourhood_method

        if clustering_method not in ['k-means', 'qr-clustering']:
            raise ValueError(f'Clustering method {clustering_method} not supported, please choose from (k-means | qr)...')
        self.clustering_method: str = clustering_method

        if not eigen_vectors:
            self.eigen_vectors: int = cluster_count
        else:
            self.eigen_vectors: int = eigen_vectors
    

    def _compute_connectivity_matrix(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Compute the connectivity matrix for the dataset. The connectivity is a binary value based on
        the distance between datapoints. The connectivity matrix is not necessarily symmetric, especially
        in the case when the knn-method is used for its computation.
        '''

        distances = self.metric(data, data, **self.metric_kwargs)

        if self.neighbourhood_method == 'epsilon':
            connectivity = np.where(distances < self.epsilon, 1, 0)
            np.fill_diagonal(connectivity, val = 0)

            return connectivity

        elif self.neighbourhood_method == 'knn':
            nearest_indices = np.argpartition(distances, kth = self.neighbour_count + 1, axis = 0)[:self.neighbour_count + 1]

            connectivity = np.zeros(shape = distances.shape)
            connectivity[np.arange(connectivity.shape[0]), nearest_indices] = 1
            np.fill_diagonal(connectivity, val = 0)

            return connectivity
        
        else:
            raise ValueError(f'The provided method for determing the neighbourhoods ({self.neighbourhood_method}) is not supported...')
    

    def _construct_laplacian_matrix(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Construct the normalized graph Laplacian using the connectivity-matrix that is computed
        using the method defined above. The normalized version is used to reduce the influence
        of vertices with large degrees.
        '''

        connectivity = self._compute_connectivity_matrix(data = data)
        ## Convert the connectivity matrix into a symmetric adjacency matrix. There is no need to
        ## worry about the diagonal terms in the adjacency matrix as they are set to zero in the
        ## construction process anyway.
        adjacency = .5 * (connectivity + connectivity.T)
        ## Compute the degree of the adjacency matrix.
        degree = np.diag(np.nansum(adjacency, axis = 1))
        inv_sqrt_degree = np.linalg.inv(np.sqrt(degree))

        laplacian = np.eye(N = data.shape[0]) - inv_sqrt_degree @ adjacency @ inv_sqrt_degree

        return laplacian
    

    def _spectral_embedding(self, laplacian: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Perform the spectral embedding of the Laplacian matrix, i.e. compute the eigen vectors of
        the Laplacian corresponding to the k (= self.eigen_vectors) smallest eigenvalues.
        '''
        return eigs(A = laplacian, k = self.eigen_vectors, which = 'SM', return_eigenvectors = True)[1]
    

    def qr_cluster(self, mapping: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        '''
        Perform the clustering using QR-decomposition and the SVD to find a discrete partitioning
        close to the eigenvector embedding as described by Damle and colleagues 2019.
        '''

        components = mapping.shape[1]
        _, _, pivot = qr(a = mapping.T, pivoting = True, mode = 'full') # type: ignore
        ut, _, v = svd(mapping[pivot[: components], :].T)
        vectors = np.abs(np.dot(mapping, np.dot(ut, v.conj())))
        return np.argmax(vectors, axis = 1)
    

    def fit(self, X: npt.NDArray[np.float_], y = None) -> None:
        '''
        This method performs the actual clustering following the steps of the spectral clustering
        algorithm. It constructs the Laplacian matrix and computes its eigenvalues and vectors.
        These can then be clustered either using a k-means clustering run or the QR-clustering
        method described by Damle et al. 2019, (https://doi.org/10.1093/imaiai/iay008).
        '''

        laplacian = self._construct_laplacian_matrix(data = X)
        mapping = self._spectral_embedding(laplacian = laplacian)

        if self.clustering_method == 'k-means':
            kmeans = KMeans(cluster_count = self.cluster_count, distance_metric = self.metric, distance_metric_kwargs = self.metric_kwargs)
            self.labels = kmeans.fit_predict(X = mapping)

        else:
            self.labels = self.qr_cluster(mapping = mapping)


    def fit_predict(self, X: npt.NDArray[np.float_], y = None) -> npt.NDArray[np.int_]:
        '''
        This is a convenience method that performs the clustering first and also returns the
        inferred cluster labels.
        '''

        self.fit(X = X)
        return self.labels