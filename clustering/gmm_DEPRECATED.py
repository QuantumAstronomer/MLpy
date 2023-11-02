
import copy as cp
import numpy as np

from numpy import typing as npt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from typing import Literal

from .kmeans import KMeans
from ..metrics import euclidean

## TO DO: write documentation and comments in the code

class GMM():

    def __init__(self, cluster_count: int, 
                 max_iteration: int = 30, 
                 tolerance: float = 1e-4,
                 covariance_regularization: float = 1e-6,
                 initialization_method: Literal['k-means++', 'k-means', 'random', 'random-data'] = 'k-means++') -> None:

        self.cluster_count             = cluster_count
        self.max_iteration             = max_iteration
        self.tolerance                 = tolerance
        self.covariance_regularization = covariance_regularization

        if initialization_method not in ['k-means++', 'k-means', 'random', 'random-data']:
            raise ValueError(f'Provided initialization method ({initialization_method}) not supported.')
        
        self.initialization_method: str = initialization_method

    
    def kmeanspp(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:

        indices = np.random.choice(data.shape[0], size = 1)
        initial_centroids = data[indices]

        for i in range(1, self.cluster_count):
            distances = np.min(euclidean(data, initial_centroids), axis = 1)**2
            probabilities = distances / np.sum(distances)
            next_index = np.random.choice(data.shape[0], size = 1, p = probabilities)
            next_centroid = data[next_index]
            indices = np.vstack([indices, next_index])
            initial_centroids = np.vstack([initial_centroids, next_centroid])
        return indices


    def initialize_responsibilities(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:

        self.shape = data.shape
        self.sample_count, self.feature_count = self.shape

        if self.initialization_method == 'random':
            responsibilities = np.random.uniform(size = (self.sample_count, self.cluster_count))
            responsibilities /= np.nansum(responsibilities, axis = 1, keepdims = True)

        elif self.initialization_method == 'random-data':
            responsibilities = np.zeros(shape = (self.sample_count, self.cluster_count))
            indices = np.random.choice(a = self.sample_count, size = self.cluster_count, replace = False)
            responsibilities[indices, np.arange(self.cluster_count)] = 1

        elif self.initialization_method == 'k-means':
            responsibilities = np.zeros(shape = (self.sample_count, self.cluster_count))
            labels = KMeans(cluster_count = self.cluster_count, initialization_count = 1, algorithm = 'hartigan-wong', max_iteration = 3).fit_predict(data = data)
            responsibilities[np.arange(self.sample_count), labels] = 1

        elif self.initialization_method == 'k-means++':
            responsibilities = np.zeros(shape = (self.sample_count, self.cluster_count))
            indices = self.kmeanspp(data = data).flatten()
            responsibilities[indices, np.arange(self.cluster_count)] = 1

        else:
            raise ValueError(f'Provided initialization method ({self.initialization_method}) not supported.')
        
        return responsibilities


    def _estimate_gaussian_paramters(self, data: npt.NDArray[np.float_], responsibilities: npt.NDArray[np.float_ | np.int_]):

        weights = np.nansum(responsibilities, axis = 0) + 10 * np.finfo(np.float64).eps

        means = np.dot(responsibilities.T, data) / weights[:, np.newaxis]
        covariances = np.zeros(shape = (self.cluster_count, self.feature_count, self.feature_count))

        for k in range(self.cluster_count):
            difference = data - means[k]
            covariances[k] = np.dot(responsibilities[:, k] * difference.T, difference) / weights[k]
            covariances[k] += self.covariance_regularization * np.eye(self.feature_count)

        return weights, means, covariances
    

    def initialize(self, data: npt.NDArray[np.float_]) -> None:

        responsibilities = self.initialize_responsibilities(data = data)

        self.weights, self.means, self.covariances = self._estimate_gaussian_paramters(data = data, responsibilities = responsibilities)
        self.weights /= np.nansum(self.weights, keepdims = True)


    def _log_probabilities(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:

        likelihood = np.zeros(shape = (self.sample_count, self.cluster_count))

        for k in range(self.cluster_count):
            distribution = multivariate_normal(mean = self.means[k], cov = self.covariances[k] + 
                                               self.covariance_regularization * np.eye(N = self.feature_count))
            likelihood[:, k] = distribution.logpdf(x = data)
        return likelihood
    
    def _log_weights(self) -> npt.NDArray[np.float_]:
        return np.log(self.weights)
    
    def _log_weighted_probabilities(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return self._log_probabilities(data = data) + self._log_weights()
    
    def _log_probability_responsibilities(self, data: npt.NDArray[np.float_]) -> tuple[npt.NDArray, npt.NDArray]:

        weighted_log_prob = self._log_weighted_probabilities(data = data)
        log_normalised_probabilities = np.array(logsumexp(weighted_log_prob, axis = 1, keepdims = True))

        log_responsibilities = weighted_log_prob - log_normalised_probabilities
        return log_normalised_probabilities, log_responsibilities


    def expectation(self, data: npt.NDArray[np.float_]) -> tuple[np.float_, npt.NDArray]:
        
        log_normalised_probabilities, log_responsibilities = self._log_probability_responsibilities(data = data)
        return np.nanmean(log_normalised_probabilities), log_responsibilities


    def maximization(self, data: npt.NDArray[np.float_], log_responsibilities: npt.NDArray[np.float_]) -> None:
        
        self.weights, self.means, self.covariances = self._estimate_gaussian_paramters(data = data, responsibilities = np.exp(log_responsibilities))
        self.weights /= self.weights.sum()

    
    def fit(self, data: npt.NDArray[np.float_]) -> None:
        
        self.initialize(data = data)

        for k in range(self.max_iteration):
            
            previous_means = cp.deepcopy(self.means)
            previous_covariances = cp.deepcopy(self.covariances)
            log_prob_norm, log_responsibilities = self.expectation(data = data)
            self.maximization(data = data, log_responsibilities = log_responsibilities)

            if np.all(np.linalg.norm((self.means - previous_means) / previous_means, axis = 1) < self.tolerance) & \
            np.all(np.linalg.norm((self.covariances - previous_covariances) / previous_covariances, axis = (1, 2)) < self.tolerance):
                break

            self.convergence_iterations = k + 1


    def fit_predict(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:

        self.fit(data = data)
        return np.argmax(self.expectation(data = data)[1], axis = 1)