import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)  # choose K points out of N
    centers = features[idxs]  # these K points are the initial centers
    assignments = np.zeros(N)

    for n in range(num_iters):

        ### YOUR CODE HERE##############################################################################################

        dist_matrix = cdist(features, centers, metric='euclidean')

        # Compute distance to each center; Assign closest center_idx
        for i in range(N):
            min_idx = np.argmin(dist_matrix[i])
            assignments[i] = min_idx

        # Shift the centers according to the mean of the assigned points
        for j in range(k):
            assigned = np.where(assignments==j)[0]
            mean = np.sum(features[assigned], axis=0) / len(assigned)
            centers[j] = mean

        ### END YOUR CODE###############################################################################################

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    ### YOUR CODE HERE##################################################################################################

    last_assignments = assignments.copy()

    for n in range(num_iters):

        dist_matrix = cdist(features, centers, metric='euclidean')

        # Compute distance to each center; Assign closest center_idx
        for i in range(N):
            min_idx = np.argmin(dist_matrix[i])
            assignments[i] = min_idx

        # Shift the centers according to the mean of the assigned points
        for j in range(k):
            assigned = np.where(assignments == j)[0]
            if len(assigned) != 0:

                mean = np.mean(features[assigned], axis=0)
                centers[j] = mean

        # if assignments didn't change, break the loop
        if np.array_equal(last_assignments, assignments):
            break

        last_assignments = assignments

        ### END YOUR CODE###############################################################################################

    return assignments


def kmeans_mini_batch(features, k, num_iters=100, batch_percent=0.8):
    """
    Try mini-batch Kmeans algorithm.

    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]

    ### YOUR CODE HERE##################################################################################################

    batch_size = int(N * batch_percent)
    assignments = np.zeros(batch_size)

    for n in range(num_iters):

        sample_idx = np.random.choice(N, size=batch_size, replace=False)

        sample_features = features[sample_idx]

        dist_matrix = cdist(sample_features, centers, metric='euclidean')

        # Compute distance to each center; Assign closest center_idx
        for i in range(batch_size):
            min_idx = np.argmin(dist_matrix[i])
            assignments[i] = min_idx

        # Shift the centers according to the mean of the assigned points
        for j in range(k):
            assigned = np.where(assignments == j)[0]
            if len(assigned) != 0:
                mean = np.mean(features[assigned], axis=0)
                centers[j] = mean

    # Full-batch assignment
    assignments = np.zeros(N)
    dist_matrix = cdist(features, centers, metric='euclidean')
    for i in range(N):
        min_idx = np.argmin(dist_matrix[i])
        assignments[i] = min_idx

        ### END YOUR CODE###############################################################################################

    return assignments




def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

        ### YOUR CODE HERE##############################################################################################

    while n_clusters > k:
        dist_matrix = cdist(centers, centers, metric='euclidean')
        # replace 0 values with maximal values
        dist_matrix[dist_matrix==0] = np.max(dist_matrix)

        # ith and jth points are the closest pair
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        new_center = np.mean(centers[(i, j),:], axis=0)

        # find data in a same group
        idx_i = np.where(assignments==i)[0]
        idx_j = np.where(assignments==j)[0]

        # assign same value if all data in a same group
        centers[idx_i] = new_center
        centers[idx_j] = new_center
        assignments[idx_i] = i
        assignments[idx_j] = i

        # each iter only group 1 Pair of data (only 2 data once)
        # so the # of center decrease 1
        n_clusters -= 1

    # revalue the assignment values to 0~k
    # e.g. from[0...125...341...435] --> [0...1...2...3]
    for i in range(k):
        for j in range(N):
            if assignments[j]>i:
                assignments[assignments==assignments[j]] = i+1
                break

        ### END YOUR CODE###############################################################################################

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color. Does Not include Position information

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE##################################################################################################

    features = img.reshape(-1, C)

    ### END YOUR CODE###################################################################################################

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful  (np.d/h/v stack: axis=2,1,0)
    - You may use np.mean and np.std --> for normalization

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE##################################################################################################

    features[:,:C] = color.reshape((-1,C))  # construct the (r,g,b,.,.) vector
    features[:,C] = np.mgrid[:H, :W][0].reshape((H*W))  # (r,g,b,x,.) vector
    features[:,C+1] = np.mgrid[:H, :W][1].reshape((H*W))  # # (r,g,b,x,y) vector

    # normalize each feature (axis=0, #H*W of (r,g,b,x,y))
    features -= np.mean(features, axis=0)
    features /= np.std(features, axis=0)

    ### END YOUR CODE###################################################################################################

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features
    

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

        Task: Compare 2 matrices with bunch of 0 and 1, check how well they match.
        Accuracy: (TP+TN)/(P+N)  == (TruePosi+TrueNega)
    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.

            e.g:         Parts of Cat:1   ;  Parts of Background:0

        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE##################################################################################################

    num_trueposi = np.sum(np.logical_and(mask_gt, mask)

    num_truenega = np.sum(np.logical_and(1-mask_gt, 1-mask)  # by (1-mask), now 1->0 and 0->1

    accuracy = float(num_trueposi + num_truenega) / np.size(mask)

    ### END YOUR CODE###################################################################################################

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
