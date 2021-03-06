import numpy as np
from scipy import stats

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0), (pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE

    # always filp the kernel first before performing Convolution
    kernel_flip = np.flip(kernel, axis=0)
    kernel_flip = np.flip(kernel_flip, axis=1)

    for i in range(Hi):
        for j in range(Wi):
            curr_window = padded[i:i+Hk, j:j+Wk]
            conv_value = np.sum(curr_window * kernel_flip)
            out[i, j] = conv_value
        if ((100*i/Hi)%50 == 0) and ((100*i/Hi) != 0):
            print('Completed %d' % (100*i/Hi) + '%')

    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))

    ### YOUR CODE HERE

    var = np.square(sigma)

    if size%2 == 0:
        return 'Please define a kernel of size 2n, not 2n+1'

    else:
        k = (size-1)/2
        for i in range(size):
            for j in range(size):
                value = (1/(2*np.pi*var)) * np.exp((-1/(2*var)) *  ((i-k)**2+(j-k)**2))
                kernel[i, j] = value

    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE

    xkernel = np.array([[1/2, 0, -1/2]])
    out = conv(img, xkernel)

    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE

    ykernel = np.array([[1/2, 0, -1/2]]).reshape((3, 1))
    out = conv(img, ykernel)

    ### END YOUR CODE

    return out

def gradient(img, degree=True):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE

    gx = partial_x(img)
    gy = partial_y(img)

    # directly perform numpy element-wise operation.
    SSG = (gx**2) + (gy**2)
    G = np.sqrt(SSG)

    if degree==True:
        # computer theta as +/-180 degree
        theta = np.arctan2(gy, gx) * 180 / np.pi
        return G, theta

    else:
        theta = np.arctan2(gy, gx)
        return G, theta

    ### END YOUR CODE


def non_maximum_suppression(G, theta, nms_size=1):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE

    for i in range(1, H-1):
        for j in range(1, W-1):

            curr_theta = theta[i, j]

            if nms_size == 1:
                # theta==45/-135 are in the Same direction
                if (curr_theta == 0) or (curr_theta == 180) or (curr_theta == -180):
                    if (G[i, j] >= G[i+1, j]) and (G[i, j] >= G[i-1, j]):
                        out[i, j] = G[i, j]
                if (curr_theta == 45) or (curr_theta == -135):
                    if (G[i, j] >= G[i+1, j+1]) and (G[i, j] >= G[i-1, j-1]):
                        out[i, j] = G[i, j]
                if (curr_theta == 90) or (curr_theta == -90):
                    if (G[i, j] >= G[i, j+1]) and (G[i, j] >= G[i, j-1]):
                        out[i, j] = G[i, j]
                if (curr_theta == 135) or (curr_theta == -45):
                    if (G[i, j] >= G[i+1, j-1]) and (G[i, j] >= G[i-1, j+1]):
                        out[i, j] = G[i, j]

            if nms_size == 2:
                # The magnitude should be greater than its neighbors in the 5x5 window
                if (i>1) and (i<(H-2)) and (j>1) and (j<(W-2)):
                    if (curr_theta == 0) or (curr_theta == 180) or (curr_theta == -180):
                        if (G[i, j] >= G[i + 1, j]) and (G[i, j] >= G[i - 1, j])\
                                and (G[i, j] >= G[i + 2, j]) and (G[i, j] >= G[i - 2, j]):
                            out[i, j] = G[i, j]
                    if (curr_theta == 45) or (curr_theta == -135):
                        if (G[i, j] >= G[i + 1, j + 1]) and (G[i, j] >= G[i - 1, j - 1])\
                                and (G[i, j] >= G[i + 2, j + 2]) and (G[i, j] >= G[i - 2, j - 2]):
                            out[i, j] = G[i, j]
                    if (curr_theta == 90) or (curr_theta == -90):
                        if (G[i, j] >= G[i, j + 1]) and (G[i, j] >= G[i, j - 1])\
                                and (G[i, j] >= G[i, j + 2]) and (G[i, j] >= G[i, j - 2]):
                            out[i, j] = G[i, j]
                    if (curr_theta == 135) or (curr_theta == -45):
                        if (G[i, j] >= G[i + 1, j - 1]) and (G[i, j] >= G[i - 1, j + 1])\
                                and (G[i, j] >= G[i + 2, j - 2]) and (G[i, j] >= G[i - 2, j + 2]):
                            out[i, j] = G[i, j]

    ### END YOUR CODE
    return out


def double_thresholding(nms, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """
    ### YOUR CODE HERE

    strong_edges = nms.copy()
    strong_edges[strong_edges < high] = 0

    weak_edges = nms.copy()
    weak_edges[np.logical_or(weak_edges > high, weak_edges < low)] = 0

    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape

    # find out all Non-zero indices and Stack as a np.array.T  (N rows and 2 cols(y, x))
    indices = np.stack(np.nonzero(strong_edges)).T

    edges = np.zeros((H, W))

    ### YOUR CODE HERE

    edges = strong_edges.copy()

    for i in range(len(indices)):
        y, x = indices[i]
        neighbor = get_neighbors(y, x, H, W)

        for j in range(len(neighbor)):
            if weak_edges[neighbor[j]] != 0:
                edges[neighbor[j]] = weak_edges[neighbor[j]]
                ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15, nms_size=1):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE

    kernel = gaussian_kernel(kernel_size, sigma)

    blurred_img = conv(img, kernel)

    grad_mag, grad_theta = gradient(blurred_img)

    nms = non_maximum_suppression(grad_mag, grad_theta, nms_size)

    strong_edges, weak_edges = double_thresholding(nms, high, low)

    edge = link_edges(strong_edges, weak_edges )


    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image(0,1) of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    # longest_distance --> the diagonal length of img
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    # from 90 to 1/2 pi (== 1.57)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)  # record the indices of those nonzeros in img(0,1)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE

    for i in range(len(ys)):
        for j in range(num_thetas):
            # compute rho = x*cos + y*sin

            rho = np.int_(np.rint(xs[i]*cos_t[j] + ys[i]*sin_t[j]))
            rho_idx = rho + diag_len  # FIND the right index in accumulator!
            accumulator[rho_idx, j] += 1

    ### END YOUR CODE

    return accumulator, rhos, thetas
