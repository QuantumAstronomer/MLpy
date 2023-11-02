import copy as cp
import numpy as np

from numpy import typing as npt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from typing import Literal

from .kmeans import KMeans
from ..metrics import euclidean


class GMM():

    '''
    Class to perform clustering based on the idea of a Gaussian Mixutre Model of the data. It thus
    separates data based on Gaussian distributions that are fitted to the data. This does require
    the user to specify the number of clusters that are expected in the data.

    The current implementation employs the Expectation-Maximization algorithm and uses logarithmic
    probabilities for enhanced numerical stability, i.e. in some circumstances the non-logarithmic
    probabilities will be so small and therefore rounded to zero due to machine precision in 
    the floating point numbers.

    Parameters:
    ===============================================================================================

        cluster_count (int, no default): Number of clusters to form, i.e. the number of centroids
        to generate around which the data is clustered.

        max_iteration (int, default = 20): Maximum number of iteration of the k-means algorithm in
        a single run.

        tolerance (float, default = 1e-4): Relative tolerance regarding the L2-norm of the
        difference in cluster centers of two iterations. It is used to declare convergence and
        early halting of the clustering when cluster centroids are no longer moving significantly.
        This speeds up the clustering method, reducing computation time significantly.

        covariance_regularization (float, default = 1e-6): Non-negative 'safety' parameter added
        to the diagonal elements of the covariance matrices to ensure that the covariance matrices
        are positive semi-definite at all times.

        initializtion_method ({'k-means++', 'k-means', 'random', 'random-data'}, 
                              default = 'k-means++'): 
        Method used for generating the initial means and covariances through responsibilites:
            - k-means++: Generate the responsibilities based on data points selected using the
              k-means++ algorithm. In total a number of points equal to the number of clusters
              will be selected for initialization.
            - k-means: Runs the k-means algorithm first to determine means using the centroids
              and the covariances as the within cluster covariances, i.e. responsibilities are set
              according to the resulting label of each data point.
            - random: Randomly initializes responsiblities.
            - random-data: Randomly select a number of data points equal to the cluster count as
              initial means through the responsibilities.

    '''


    def __init__(self, cluster_count: int, 
                 max_iteration: int = 30, 
                 tolerance: float = 1e-4,
                 covariance_regularization: float = 1e-6,
                 initialization_method: Literal['k-means++', 'k-means', 'random', 'random-data'] = 'k-means++') -> None:
        
        self.cluster_count: int               = cluster_count
        self.max_iteration: int               = max_iteration
        self.tolerance: float                 = tolerance
        self.covariance_regularization: float = covariance_regularization

        if initialization_method not in ['k-means++', 'k-means', 'random', 'random-data']:
            raise ValueError(f'Provided initialization method ({initialization_method}) not supported.')
        
        self.initialization_method: str = initialization_method


    def _kmeanspp(self, data: npt.NDArray[np.float_])