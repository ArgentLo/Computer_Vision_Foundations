import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    # Convolution: flip the kernel wrt x&y axises --> then Cross-Correlation
    conv_kernel = np.flip(kernel, 0)
    conv_kernel = np.flip(conv_kernel, 1)

    for i in range(int(Hk/2), Hi-int(Hk/2)-1):
        for j in range(int(Wk/2), Wi-int(Wk/2)-1):
            curr_window = image.copy()[i-int(Hk/2):i+int(Hk/2)+1, j-int(Wk/2):j+int(Wk/2)+1]
            conv = curr_window * conv_kernel
            s = np.sum(conv)
            out[i, j] = s

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE

    m = H + (2 * int(pad_height))
    n = W + (2 * int(pad_width))

    out = np.zeros((m,n))

    out[int(pad_height):(int(pad_height)+H), int(pad_width):(int(pad_width)+W)] = image


    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    curr_window = np.zeros((Hk, Wk))
    conv_kernel = np.flip(kernel, 0)
    conv_kernel = np.flip(conv_kernel, 1)

    for i in range(1, Hi-1):
        for j in range(1, Wi-1):
            curr_window = image.copy()[i-1:i+2, j-1:j+2]
            conv = curr_window * conv_kernel
            s = np.sum(conv)
            out[i, j] = s

    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE


    Hi, Wi = f.shape
    Hk, Wk = g.shape
    curr_window = np.zeros((Hk, Wk))
    out = np.zeros((Hi, Wi))

    for i in range(int(Hk/2), Hi-int(Hk/2), int(Hk/4)):
        for j in range(int(Wk/2), Wi-int(Wk/2), int(Wk/4)):
            curr_window = f.copy()[i-int(Hk/2):i+int(Hk/2), j-int(Wk/2):j+int(Wk/2)+1]
            cross_corr = curr_window * g
            s = np.sum(cross_corr)
            out[i, j] = s

        print('Completed %d' % (100 * (i-int(Hk/2))/int(Hi-Hk)) + '%')

    ### END YOUR CODE

    return out



def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    out = None
    ### YOUR CODE HERE

    # first, simple zero_padding
    pad_f = zero_pad(f, 30, 30)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    # compute zero_mean_template
    mean_g = np.sum(g) / (Hk * Wk)
    norm_g = g - mean_g

    for i in range(0, Hi):
        for j in range(0, Wi):

            curr_window = pad_f.copy()[i+30-int(Hk/2):i+30+int(Hk/2), j+30-int(Wk/2):j+30+int(Wk/2)+1]
            cross_corr = curr_window * norm_g
            s = np.sum(cross_corr)
            out[i, j] = s

        if (100*i/Hi)%5 == 0:
            print('Completed %d' % (100*i/Hi) + '%')

    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
