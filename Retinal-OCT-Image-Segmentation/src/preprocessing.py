
import cv2


def preprocess_image(img):
    """
    Apply sequential filtering to reduce noise in OCT images.

    Filtering techniques:
    1. Median filtering
    2. Bilateral filtering
    3. Non-Local Means denoising

    Args:
        img: Input grayscale OCT image.

    Returns:
        Denoised grayscale image.
    """

    median_img = cv2.medianBlur(img, 5)

    bilateral_img = cv2.bilateralFilter(
        median_img,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    nlm_img = cv2.fastNlMeansDenoising(
        bilateral_img,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    return nlm_img

