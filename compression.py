import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size

    u, s, vh = np.linalg.svd(image, full_matrices=False)
    compressed_u = u[:, :num_values]
    compressed_s = s[:num_values]
    compressed_vh = vh[:num_values, :]

    compressed_image = np.dot((compressed_u * compressed_s), compressed_vh)

    compressed_size = (compressed_u.shape[0] * compressed_u.shape[1]) + compressed_s.shape[0]\
                      + (compressed_vh.shape[0] * compressed_vh.shape[1])


    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
