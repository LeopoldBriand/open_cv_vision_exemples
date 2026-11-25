import cv2

import matplotlib.pyplot as plt

def get_gray_scale_and_histogram(image):
    # Convert to gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create and display an histogram to choose right values of threshold
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )
    plt.figure(figsize=(10, 5))
    plt.plot(histogram)
    plt.title("Grayscale histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")
    plt.xlim([0, 256])
    plt.grid()
    plt.savefig("output/histogram.png")
    plt.close()

    return gray

def get_mask_from_grayscale(gray):
    blurred = cv2.GaussianBlur( # This does not change much :(
        gray,
        (5, 5),
        0
    )
    _, mask = cv2.threshold(
            blurred,
            110,
            255,
            cv2.THRESH_BINARY_INV
        )
    mask = cv2.bitwise_not(mask)
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig("output/binary_mask.png", bbox_inches="tight")
    plt.close()
    return mask

def get_contour_from_mask(image, mask):
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("No bread contour detected!")

    bread_contour = max(contours, key=cv2.contourArea)
    result = image.copy()

    cv2.drawContours(
        result,
        [bread_contour],
        -1,
        (0, 255, 0),
        10
    )

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig("output/contour_visualization.png", bbox_inches="tight")
    plt.close()

    return bread_contour
