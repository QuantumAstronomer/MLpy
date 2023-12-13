'''
This file contains the functions to calculate the pairwise distances between two arrays containing
a set of coordinates in P-dimensional space according to various different distance metrics.

The call signature of each function begins with two numpy ndarrays labeled X and Y, respectively.
Each function has the option to calculate the pairwise distances between all the coordinates pairs
in the X-array by not providing a Y array, as this has a default parameter equal to None.

If two arrays are provided these should have the same shape in the second dimension, i.e. if X has
shape N x P then Y has to be of shape M x P. The resulting array contains the distances from
x1 to y1, x1 to y2, ... x1 to ym, all the way up to xn to ym. This array thus has shape N x M.
'''


import numpy as np

from numpy import typing as npt


def euclidean(X: npt.NDArray[np.float_], Y: npt.NDArray[np.float_] | None = None) -> npt.NDArray[np.float_]:
    '''
    Compute the pairwise Euclidean distances between two arrays of coordinates. The coordinate
    arrays can have an arbitrary number of dimensions in feature-space.

    Parametrs:
    ===============================================================================================

        X, Y (numpy ndarray): Coordinate arrays containing N and M P-dimensional points for which 
        to calculate the pairwise Euclidean distances. If Y is None the function is called with
        Y = X, i.e. the distances between all the coordinates in X are calculated.

    Returns:
    ===============================================================================================

        numpy ndarray: An N x M (or N x N) array containing the Euclidean separations between each
        pair of coordinates in the input arrays.
    '''

    ## ============================================================================================ ##
    ## This is the option for when the 'self-distance' of a coordinate array is desired, it is      ##
    ## achieved by simply calling the function with the X and Y coordinate arrays being the same.   ##
    ## ============================================================================================ ##
    
    if Y is None:
        Y = X
    
    ## ============================================================================================ ##
    ## Check if both arrays are 2-dimensional such that the first dimension contains the number of  ##
    ## datapoints and the second dimension the various features. A singular dimensional array can   ##
    ## not be interpreted correctly for distance calculations. If the input is singular dimensional ##
    ## and has K entries, should it be cast in to a 1 x K or K x 1 array?                           ##
    ## ============================================================================================ ##
    
    if X.ndim != 2:
        raise ValueError(f'Coordinate array of shape {X.shape} with dimension {X.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    if Y.ndim != 2:
        raise ValueError(f'Coordinate array of shape {Y.shape} with dimension {Y.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    ## ============================================================================================ ##
    ## Check that both coordinate arrays have the same shape in the second dimension, i.e. have the ##
    ## same number of features. Otherwise it is nonsensical to try to calculate distances.          ##
    ## ============================================================================================ ##

    if not X.shape[1] == Y.shape[1]:
        raise ValueError(f'Feature dimensionaility of X ({X.shape[1]}) is not equal to the feature dimensionality '
                         f'of Y ({Y.shape[1]}). Please provide arrays of equal feature dimensionality.')
    
    return np.linalg.norm(X[:, :, None] - Y[:, :, None].T, axis = 1)


def manhattan(X: npt.NDArray[np.float_], Y: npt.NDArray[np.float_] | None = None) -> npt.NDArray[np.float_]:
    '''
    Compute the pairwise Manhattan distances between two arrays of coordinates. The coordinate
    arrays can have an arbitrary number of dimensions in feature-space.

    Parametrs:
    ===============================================================================================

        X, Y (numpy ndarray): Coordinate arrays containing N and M P-dimensional points for which 
        to calculate the pairwise Manhattan distances. If Y is None the function is called with
        Y = X, i.e. the distances between all the coordinates in X are calculated.

    Returns:
    ===============================================================================================

        numpy ndarray: An N x M (or N x N) array containing the Manhattan separations between each
        pair of coordinates in the input arrays.
    '''

    ## ============================================================================================ ##
    ## This is the option for when the 'self-distance' of a coordinate array is desired, it is      ##
    ## achieved by simply calling the function with the X and Y coordinate arrays being the same.   ##
    ## ============================================================================================ ##
    
    if Y is None:
        Y = X
    
    ## ============================================================================================ ##
    ## Check if both arrays are 2-dimensional such that the first dimension contains the number of  ##
    ## datapoints and the second dimension the various features. A singular dimensional array can   ##
    ## not be interpreted correctly for distance calculations. If the input is singular dimensional ##
    ## and has K entries, should it be cast in to a 1 x K or K x 1 array?                           ##
    ## ============================================================================================ ##
    
    if X.ndim != 2:
        raise ValueError(f'Coordinate array of shape {X.shape} with dimension {X.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    if Y.ndim != 2:
        raise ValueError(f'Coordinate array of shape {Y.shape} with dimension {Y.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    ## ============================================================================================ ##
    ## Check that both coordinate arrays have the same shape in the second dimension, i.e. have the ##
    ## same number of features. Otherwise it is nonsensical to try to calculate distances.          ##
    ## ============================================================================================ ##

    if not X.shape[1] == Y.shape[1]:
        raise ValueError(f'Feature dimensionaility of X ({X.shape[1]}) is not equal to the feature dimensionality '
                         f'of Y ({Y.shape[1]}). Please provide arrays of equal feature dimensionality.')
    
    return np.sum(np.abs(X[:, :, None] - Y[:, :, None].T), axis = 1)


def chebyshev(X: npt.NDArray[np.float_], Y: npt.NDArray[np.float_] | None = None) -> npt.NDArray[np.float_]:
    '''
    Compute the pairwise Chebyshev distances between two arrays of coordinates. The coordinate
    arrays can have an arbitrary number of dimensions in feature-space.

    Parametrs:
    ===============================================================================================

        X, Y (numpy ndarray): Coordinate arrays containing N and M P-dimensional points for which 
        to calculate the pairwise Chebyshev distances. If Y is None the function is called with
        Y = X, i.e. the distances between all the coordinates in X are calculated.

    Returns:
    ===============================================================================================

        numpy ndarray: An N x M (or N x N) array containing the Chebyshev separations between each
        pair of coordinates in the input arrays.
    '''

    ## ============================================================================================ ##
    ## This is the option for when the 'self-distance' of a coordinate array is desired, it is      ##
    ## achieved by simply calling the function with the X and Y coordinate arrays being the same.   ##
    ## ============================================================================================ ##
    
    if Y is None:
        Y = X
    
    ## ============================================================================================ ##
    ## Check if both arrays are 2-dimensional such that the first dimension contains the number of  ##
    ## datapoints and the second dimension the various features. A singular dimensional array can   ##
    ## not be interpreted correctly for distance calculations. If the input is singular dimensional ##
    ## and has K entries, should it be cast in to a 1 x K or K x 1 array?                           ##
    ## ============================================================================================ ##
    
    if X.ndim != 2:
        raise ValueError(f'Coordinate array of shape {X.shape} with dimension {X.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    if Y.ndim != 2:
        raise ValueError(f'Coordinate array of shape {Y.shape} with dimension {Y.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    ## ============================================================================================ ##
    ## Check that both coordinate arrays have the same shape in the second dimension, i.e. have the ##
    ## same number of features. Otherwise it is nonsensical to try to calculate distances.          ##
    ## ============================================================================================ ##

    if not X.shape[1] == Y.shape[1]:
        raise ValueError(f'Feature dimensionaility of X ({X.shape[1]}) is not equal to the feature dimensionality '
                         f'of Y ({Y.shape[1]}). Please provide arrays of equal feature dimensionality.')
    
    return np.max(np.abs(X[:, :, None] - Y[:, :, None].T), axis = 1)


def minkowski(X: npt.NDArray[np.float_], Y: npt.NDArray[np.float_] | None = None, p: int | float = 3) -> npt.NDArray[np.float_]:
    '''
    Compute the pairwise Minkowski distances between two arrays of coordinates. The coordinate
    arrays can have an arbitrary number of dimensions in feature-space.

    Parametrs:
    ===============================================================================================

        X, Y (numpy ndarray): Coordinate arrays containing N and M P-dimensional points for which 
        to calculate the pairwise Minkowski distances. If Y is None the function is called with
        Y = X, i.e. the distances between all the coordinates in X are calculated.

        p (int or float): Order to be used in the Minkowski distance metric. p should be a positive
        floating point or integer number. If p = 1 Minkowski is equal to Manhattan, p = 2 is equal
        to the Euclidean distance, and p = infinity equates to the chebyshev distance. In these
        cases the other functions should be used for computational efficiency.

    Returns:
    ===============================================================================================

        numpy ndarray: An N x M (or N x N) array containing the Minkowski separations between each
        pair of coordinates in the input arrays.
    '''

    ## ============================================================================================ ##
    ## This is the option for when the 'self-distance' of a coordinate array is desired, it is      ##
    ## achieved by simply calling the function with the X and Y coordinate arrays being the same.   ##
    ## ============================================================================================ ##
    
    if Y is None:
        Y = X
    
    ## ============================================================================================ ##
    ## Check if both arrays are 2-dimensional such that the first dimension contains the number of  ##
    ## datapoints and the second dimension the various features. A singular dimensional array can   ##
    ## not be interpreted correctly for distance calculations. If the input is singular dimensional ##
    ## and has K entries, should it be cast in to a 1 x K or K x 1 array?                           ##
    ## ============================================================================================ ##
    
    if X.ndim != 2:
        raise ValueError(f'Coordinate array of shape {X.shape} with dimension {X.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    if Y.ndim != 2:
        raise ValueError(f'Coordinate array of shape {Y.shape} with dimension {Y.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    ## ============================================================================================ ##
    ## Check that both coordinate arrays have the same shape in the second dimension, i.e. have the ##
    ## same number of features. Otherwise it is nonsensical to try to calculate distances.          ##
    ## ============================================================================================ ##

    if not X.shape[1] == Y.shape[1]:
        raise ValueError(f'Feature dimensionaility of X ({X.shape[1]}) is not equal to the feature dimensionality '
                         f'of Y ({Y.shape[1]}). Please provide arrays of equal feature dimensionality.')
    
    return np.float_power(np.sum(np.float_power(np.abs(X[:, :, None] - Y[:, :, None].T), p)), 1/ p )


def cosine(X: npt.NDArray[np.float_], Y: npt.NDArray[np.float_] | None = None) -> npt.NDArray[np.float_]:
    '''
    Compute the pairwise Cosine distances between two arrays of coordinates. The coordinate
    arrays can have an arbitrary number of dimensions in feature-space. Note that, unlike the
    previous distance metrics, the cosine distance is a dimensionless metric and varies between
    -1 and 1.

    Parametrs:
    ===============================================================================================

        X, Y (numpy ndarray): Coordinate arrays containing N and M P-dimensional points for which 
        to calculate the pairwise Cosine distances. If Y is None the function is called with
        Y = X, i.e. the distances between all the coordinates in X are calculated.

    Returns:
    ===============================================================================================
    
        numpy ndarray: An N x M (or N x N) array containing the Cosine separations between each
        pair of coordinates in the input arrays.
    '''

    ## ============================================================================================ ##
    ## This is the option for when the 'self-distance' of a coordinate array is desired, it is      ##
    ## achieved by simply calling the function with the X and Y coordinate arrays being the same.   ##
    ## ============================================================================================ ##
    
    if Y is None:
        Y = X
    
    ## ============================================================================================ ##
    ## Check if both arrays are 2-dimensional such that the first dimension contains the number of  ##
    ## datapoints and the second dimension the various features. A singular dimensional array can   ##
    ## not be interpreted correctly for distance calculations. If the input is singular dimensional ##
    ## and has K entries, should it be cast in to a 1 x K or K x 1 array?                           ##
    ## ============================================================================================ ##
    
    if X.ndim != 2:
        raise ValueError(f'Coordinate array of shape {X.shape} with dimension {X.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    if Y.ndim != 2:
        raise ValueError(f'Coordinate array of shape {Y.shape} with dimension {Y.ndim} can not be interpreted '
                         f'correctly to be used for distance calculations. Please provide an array with 2 dimensions.')
    
    ## ============================================================================================ ##
    ## Check that both coordinate arrays have the same shape in the second dimension, i.e. have the ##
    ## same number of features. Otherwise it is nonsensical to try to calculate distances.          ##
    ## ============================================================================================ ##

    if not X.shape[1] == Y.shape[1]:
        raise ValueError(f'Feature dimensionaility of X ({X.shape[1]}) is not equal to the feature dimensionality '
                         f'of Y ({Y.shape[1]}). Please provide arrays of equal feature dimensionality.')
    
    return 1 - np.dot(X, Y.T) / np.outer(np.linalg.norm(X, axis = 1), np.linalg.norm(Y, axis = 1))