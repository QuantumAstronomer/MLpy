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

        covariance_type ({'full', 'tied', 'diagonal', 'spherical'}, default = 'full'):
        Type of covariances to use for the components, i.e. whether they are fully free or 
        restricted in some way.

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
                 covariance_type: Literal['full', 'tied', 'diagonal', 'spherical'] = 'full',
                 initialization_method: Literal['k-means++', 'k-means', 'random', 'random-data'] = 'k-means++') -> None:
        
        self.cluster_count: int               = cluster_count
        self.max_iteration: int               = max_iteration
        self.tolerance: float                 = tolerance
        self.covariance_regularization: float = covariance_regularization

        if covariance_type not in ['full', 'tied', 'diagonal', 'spherical']:
            raise ValueError(f'Provided covariance type ({covariance_type}) is not supported...')
        
        self.covariance_type: str = covariance_type

        if initialization_method not in ['k-means++', 'k-means', 'random', 'random-data']:
            raise ValueError(f'Provided initialization method ({initialization_method}) not supported.')
        
        self.initialization_method: str = initialization_method


    def _kmeanspp(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        '''
        Use the k-means++ method to initialize the responsibilities used for determining the
        initial means and covariance matrices. This implementation is slightly different from the
        one provided in the k-means module as it returns the indices of the datapoints chosen to be
        the initial means.
        '''

        ## Pick the first mean/datapoint for the responsibilities completely at random.
        index = np.random.choice(a = data.shape[0], size = 1)
        means = data[index]

        for k in range(1, self.cluster_count):

            ## Compute distance (squared) to the nearest already picked centroid for each datapoint
            ## No need to mask datapoints that are centroids as their probability will be set to
            ## zero as their minimum distance is zero.
            distances = np.nanmin(euclidean(data, means), axis = 1)**2

            ## Turn the distances in to proper probabilities by making them sum to one.
            probabilities = distances / np.nansum(distances)

            ## And pick the new index/data point at random.
            next_index = np.random.choice(data.shape[0], size = 1, p = probabilities)
            next_mean = data[next_index]
            index = np.vstack([index, next_index])
            means = np.vstack([means, next_mean])

        return index
    

    def initialize(self, data: npt.NDArray[np.float_]) -> None:
        '''
        Initialize the weights, means, and covariances for the GMM model to have a starting point.
        This is done using the responsibilities which are in turn used to initialize the parameters
        of the constituent Gaussian distributions.
        '''

        self.shape = data.shape
        self.sample_count, self.feature_count = self.shape

        if self.sample_count < self.cluster_count:
            raise ValueError(f'Number of clusters to form ({self.cluster_count}) exceeds number of data samples({self.sample_count})...'
                             f'Reduce the number of clusters or provide additional datapoints.')
        
        responsiblities = self._initialize_responsibilities(data = data)

        self.weights, self.means, self.covariances = self._estimate_gaussian_params(data = data, responsibilities = responsiblities)
        self.weights /= np.nansum(self.weights, keepdims = True)
    

    def _initialize_responsibilities(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Initialize the responsibilites from the data using the chosen initialization method, i.e.
        the one specified when initializing the Gaussian Mixture instance.

        Responsibilities are basically measures of the probability that a given datapoint belongs
        to a specific cluster.
        '''
        
        if self.initialization_method == 'random':
            ## Initialize the distribution of responsibilities completely at random.
            responsibilities = np.random.uniform(size = (self.sample_count, self.cluster_count))
            responsibilities /= np.nansum(responsibilities, axis = 1, keepdims = True)

            return responsibilities

        elif self.initialization_method == 'random-data':
            ## Choose random responsibilities using the data, i.e. pick a few random datapoints.
            responsibilities = np.zeros(shape = (self.sample_count, self.cluster_count))
            indices = np.random.choice(a = self.sample_count, size = self.cluster_count, replace = False)
            responsibilities[indices, np.arange(self.cluster_count)] = 1

            return responsibilities

        elif self.initialization_method == 'k-means++':
            ## Similar to random-data. However, now a tactic is applied for picking the datapoints
            ## less randomly to increase their spread.
            responsibilities = np.zeros(shape = (self.sample_count, self.cluster_count))
            indices = self._kmeanspp(data = data).flatten()
            responsibilities[indices, np.arange(self.cluster_count)] = 1

            return responsibilities

        elif self.initialization_method == 'k-means':
            ## Determine the initial responsibilities using a simplified k-means clustering run.
            ## While this gives the most accurate starting point, it is also computationally
            ## the most expensive option.
            responsibilities = np.zeros(shape = (self.sample_count, self.cluster_count))
            labels = KMeans(cluster_count = self.cluster_count, initialization_count = 1, algorithm = 'hartigan-wong', max_iteration = 3).fit_predict(data = data)
            responsibilities[np.arange(self.sample_count), labels] = 1

            return responsibilities
        
        else:
            raise ValueError(f'Provided initialization method ({self.initialization_method}) not supported.')

    
    def _estimate_gaussian_params(self, data: npt.NDArray[np.float_], responsibilities: npt.NDArray[np.float_])\
                                  -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        '''
        Given the data and the responsibilities, estimate the parameters of the Gaussian 
        distributions. Covariances can be determined in a few ways, depending on the user input
        chosen, i.e. full, tied, spherical, or diagonal.
        '''

        ## The weights are estimate and an error term is added so they are never zero, this is to
        ## prevent a division by zero error.
        weights = np.nansum(responsibilities, axis = 0) + 10 * np.finfo(np.float64).eps

        means: npt.NDArray[np.float_] = np.dot(responsibilities.T, data) / weights[:, np.newaxis]

        ## Estimate the covariance given the type of covariances to be used.

        if self.covariance_type == 'full':
            
            ## In full covariances, each component has its own covariance matrix which is
            ## completely independent from the other components.

            covariances = np.empty(shape = (self.cluster_count, self.feature_count, self.feature_count), dtype = np.float64)

            for k in range(self.cluster_count):
                differences = data - means[k]
                covariances[k] = np.dot(responsibilities[:, k] * differences.T, differences) / weights[k]
                covariances[k] += self.covariance_regularization * np.eye(self.feature_count)

            return weights, means, covariances

        if self.covariance_type == 'tied':

            ## In the tied covariances, the components all share the same covariance matrix. Which
            ## is simply equal to the covariance within the data sample.
            covariances = np.cov(data.T)
            covariances += self.covariance_regularization * np.eye(self.feature_count)

            return weights, means, covariances

        if self.covariance_type == 'diagonal':

            ## In diagonal covariances, each component has its own independent covariance matrix.
            ## However, it is restricted to be diagonal only.

            average_data_sq = np.dot(responsibilities.T, data * data) / weights[:, np.newaxis]
            average_means_sq = means**2
            average_data_means = means * np.dot(responsibilities.T, data) / weights[:, np.newaxis]
            covariances = average_data_sq - 2 * average_data_means + average_means_sq + self.covariance_regularization

            return weights, means, covariances
        
        if self.covariance_type == 'spherical':

            ## And finally for spherical, it is similar to diagonal: each component has its own
            ## independent covariance matrix. However, this time it is characterised by a scalar
            ## covariance leading to a spherical distribution in feature space.

            average_data_sq = np.dot(responsibilities.T, data * data) / weights[:, np.newaxis]
            average_means_sq = means**2
            average_data_means = means * np.dot(responsibilities.T, data) / weights[:, np.newaxis]
            covariances = np.nanmean(average_data_sq - 2 * average_data_means + average_means_sq, axis = 1) + self.covariance_regularization

            return weights, means, covariances
        
        else:
            raise ValueError(f'Provided covariance type ({self.covariance_type}) is not supported...')
        

    def _log_prob(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Estimate the logarithmic probabilities, i.e. log responsibilities for the data sample given
        the current gaussian parameters. These are not normalised, however.
        '''

        likelihood = np.zeros(shape = (self.sample_count, self.cluster_count))

        for k in range(self.cluster_count):
            if self.covariance_type in ['diagonal', 'spherical']:
                distribution = multivariate_normal(mean = self.means[k], cov = self.covariances[k] * np.eye(self.feature_count))
            if self.covariance_type == 'full':
                distribution = multivariate_normal(mean = self.means[k], cov = self.covariances[k])
            else:
                distribution = multivariate_normal(mean = self.means[k], cov = self.covariances)

            likelihood[:, k] = distribution.logpdf(x = data)
        return likelihood
    

    def _log_weights(self) -> npt.NDArray[np.float_]:
        '''
        This is a convenience-function to quickly obtain the logarithm of the weights.
        '''
        return np.log(self.weights)
    

    def _log_weighted_probs(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Return the weighted logarithmic probabilities, i.e. the logarithmic responsibilities
        modified by the weights of each cluster. These are not normalised, however.
        '''
        return self._log_prob(data = data) + self._log_weights()
    

    def _log_prob_resp(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        Calculate the normalised weighted logarithmic probabilities, i.e. the log responsibilities.
        '''

        weighted_log_prob = self._log_weighted_probs(data = data)
        log_prob_norm = np.array(logsumexp(weighted_log_prob, axis = 1, keepdims = True))

        log_resp = weighted_log_prob - log_prob_norm
        return log_resp
    

    def expectation(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        '''
        This is the expectation step in the EM-algorithm which simply calculates the (logarithmic)
        responsibilities for all the clusters.
        '''
        return self._log_prob_resp(data = data)
    

    def maximization(self, data: npt.NDArray[np.float_], log_resp: npt.NDArray[np.float_]) -> None:
        '''
        In the maximization step the Gaussian parameters are updated, i.e. means and covariances.
        '''

        self.weights, self.means, self.covariances = self._estimate_gaussian_params(data = data, responsibilities = np.exp(log_resp))
        self.weights /= np.nansum(self.weights)


    def fit(self, data: npt.NDArray[np.float_]) -> None:
        '''
        Perform the fitting procedure given the data at hand. Convergence is claimed once the
        relative changes in the recorded means and covariances no longer exceed the tolerance
        parameter.
        '''

        ## Start by initializing the weights, means, and covariances.
        self.initialize(data = data)

        ## Then follows the main loop of the EM-algorithm.
        for i in range(self.max_iteration):

            self.convergence_iteration = i + 1

            ## Make a copy of the current means and variances in order to stop the algorithm early
            ## reducing the computational strain and not performing unneeded iterations.
            previous_means = cp.deepcopy(self.means)
            previous_covariances = cp.deepcopy(self.covariances)

            ## Perform the EM-step.
            log_resp = self.expectation(data = data)
            self.maximization(data = data, log_resp =log_resp)

            ## Stopping criterion is calculated using the L2 norm of the difference between the
            ## means and covariances of the previous step compared to the current step.
            mean_converge = np.all(np.linalg.norm((self.means - previous_means) / previous_means, axis = 1) < self.tolerance)
            covariance_converge = np.all(np.linalg.norm((self.covariances - previous_covariances) / previous_covariances, axis = (1, 2)) < self.tolerance)

            if mean_converge and covariance_converge:
                break
        

    def predict(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        '''
        Assuming the GMM object has been fitted, return the labels given the data. This can be
        called when new data is available which is assumed to fall in the same clustering, i.e.
        the new data should not change the clustering.
        '''
        return np.argmax(self.expectation(data = data), axis = 1)
    

    def fit_predict(self, data: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        '''
        Perform the fitting and prediction in a single go.
        '''
        
        self.fit(data = data)

        ## Perform a final expectation step so fit_predict results are consistent with calling
        ## fit followed by predict.
        log_resp = self.expectation(data = data)

        return np.argmax(log_resp, axis = 1)