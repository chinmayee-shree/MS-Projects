
import numpy as np


def kmeans_segmentation_basic(img, K=4, max_iters=100):
    """
    Perform basic K-Means clustering on a grayscale OCT image.

    The darkest intensity cluster is selected as a
    potential cyst region.

    Args:
        img: Preprocessed grayscale OCT image.
        K: Number of clusters.
        max_iters: Maximum number of iterations.

    Returns:
        cyst_mask_binary: Binary segmentation mask.
        segmented_img_colored: Colored cluster visualization.
        segmented_img_grayscale: Grayscale clustered image.
    """

    # Flatten image
    Z = img.reshape(-1, 1).astype(np.float32)

    # Initialize cluster centers randomly
    np.random.seed(0)
    centers = np.random.choice(
        Z.flatten(), K, replace=False
    ).reshape(K, 1)

    for i in range(max_iters):

        # Assign each pixel to nearest cluster center
        distances = np.abs(Z - centers.T)
        labels = np.argmin(distances, axis=1)

        # Update cluster centers
        new_centers = np.array([
            Z[labels == k].mean() if np.any(labels == k)
            else centers[k][0]
            for k in range(K)
        ]).reshape(K, 1)

        # Check convergence
        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    # Identify darkest cluster
    dark_cluster_label = np.argmin(centers)

    # Binary cyst candidate mask
    cyst_mask_binary = (
        labels == dark_cluster_label
    ).reshape(img.shape).astype(np.uint8) * 255

    # Grayscale segmented image
    segmented_img_grayscale = np.array(
        [centers[label][0] for label in labels],
        dtype=np.uint8
    ).reshape(img.shape)

    # Colored visualization
    segmented_img_colored = np.zeros(
        (*img.shape, 3), dtype=np.uint8
    )

    colors = np.linspace(0, 255, K).astype(np.uint8)

    for k in range(K):
        idx = labels.reshape(img.shape) == k

        if k == dark_cluster_label:
            segmented_img_colored[idx] = [255, 0, 0]

        else:
            color_val = int(colors[k])

            if k % 3 == 0:
                segmented_img_colored[idx] = [
                    color_val, color_val, 0
                ]

            elif k % 3 == 1:
                segmented_img_colored[idx] = [
                    0, color_val, color_val
                ]

            else:
                segmented_img_colored[idx] = [
                    color_val, 0, color_val
                ]

    return (
        cyst_mask_binary,
        segmented_img_colored,
        segmented_img_grayscale
    )
