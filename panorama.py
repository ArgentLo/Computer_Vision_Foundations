import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE

    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy
    dev = window_size**2

    sum_dx2 = convolve(dx2, window, mode='constant', cval=0) / dev
    sum_dy2 = convolve(dy2, window, mode='constant', cval=0) / dev
    sum_dxy = convolve(dxy, window, mode='constant', cval=0) / dev

    # taking advantage of Numpy's element-wise operation.
    det = (sum_dx2 * sum_dy2) - (sum_dxy ** 2)
    trace_2 = (sum_dx2 * sum_dy2) ** 2
    response = det - (k * trace_2)

    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    ### YOUR CODE HERE

    h, w = patch.shape
    mean = np.sum(patch)/(h+w)
    var = np.sum((patch-mean)**2)/(h+w)
    std = np.sqrt(var)

    normal_patch = (patch-mean)/std
    normal_patch = normal_patch.reshape(1,-1)
    for i in range(normal_patch.shape[1]):
        feature.append(normal_patch[0, i])
    feature = np.array(feature)

    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest --> A B distance is smaller than A C distance
    that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    print('shape of keypoints descriptor_1:', desc1.shape)
    print('shape of keypoints descriptor_2:', desc2.shape)
    print('shape of distance btw desc_1&2 is:', dists.shape)


    for i in range(N):
        idx = np.argsort(dists[i, :])[:2]
        min_1 = idx[0]
        min_2 = idx[1]
        ratio = dists[i, min_1] / dists[i, min_2]
        if ratio < threshold:
            matches = np.concatenate((matches, [i, min_1]), axis=0)


    matches = matches.reshape(-1, 2)
    matches = np.int_(matches)

    ### END YOUR CODE
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    # construct Homogeneous Coordinate !  Pad the matrix with 1
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE

    H = np.linalg.lstsq(p2, p1)[0][:p1.shape[1], :p1.shape[1]]  # the [0] is the position of the solution

    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:, -1] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix (LSE on this random set of matches)
        3. Compute inliers (error < threshold)
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers (another loop here: repeating to find the best
                                                                    model in the current sample)

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.1)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE

    for i in range(n_iters*20):

        sample_idx = np.random.choice(N, n_samples, replace=False)

        # Least Square Estimate
        H = np.linalg.lstsq(matched2[sample_idx], matched1[sample_idx])[0][:matched1.shape[1], :matched1.shape[1]]
        # Sometimes numerical issues cause least-squares to produce the last
        # column which is not exactly [0, 0, 1]
        H[:, -1] = np.array([0, 0, 1])

        # Find inliers among matched_keypoints outside the current_sample
        excl_idx = list(set(range(N)) - set(sample_idx))
        pred_value = np.dot(matched2[excl_idx], H)
        err_sq = (pred_value - matched1[excl_idx])**2
        err = np.sqrt(err_sq[:,0]+err_sq[:,1])
        inliner_idx = np.where(err <= threshold)[0]


        if inliner_idx.size != 0 and inliner_idx.size > n_inliers:

            #  Based on both the inliners and the samples, recompute the LSE
            idx = np.concatenate((inliner_idx, sample_idx))
            best_H = np.linalg.lstsq(matched2[idx], matched1[idx])[0][:matched1.shape[1], :matched1.shape[1]]
            best_H[:, -1] = np.array([0, 0, 1])
            n_inliers = inliner_idx.size
            max_inliers = idx
        i += 1


    H = best_H


    ### END YOUR CODE
    return H, matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector

    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)

    Input: a patch(or a Block) --> Output: a long 1-d vector

    Returns:
        block: 1D array of shape n_bins*((h*w)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
   
    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180  # (-90%180)=90!
                                                      # Now, no negative_theta and put into same direction.

    # Using function from skimage.util.view_as_blocks
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    cells = np.zeros((rows, cols, n_bins))


    ### YOUR CODE HERE

    # Compute histogram per cell
    # then concatenate the vectors to form a long vector for the whole block (or patch)

    theta_bins = np.int_(theta // degrees_per_bin)  # e.g. 178//20 = 8-->the 8th bin;
                                                    # 1//20 = 0-->the 0th bin

    theta_bins[theta_bins==9] = 0
    theta_cells = view_as_blocks(theta_bins, block_shape=pixels_per_cell)


    for i in range(rows):
        for j in range(cols):
            for n in range(pixels_per_cell[0]):
                for m in range(pixels_per_cell[1]):
                    
                    angle = theta_cells[i, j, n, m]
                    
                    cells[i, j, angle] += G_cells[i, j, n, m]

    block = []
    for i in range(rows):
        for j in range(cols):
            block = np.concatenate((block, cells[i,j,:]))

    ### YOUR CODE HERE
    
    return block
