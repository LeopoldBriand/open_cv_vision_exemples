import cv2
import numpy as np
import os

from bakery.process import get_contour_from_mask, get_gray_scale_and_histogram, get_mask_from_grayscale

def extract_basic_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(
        contour,
        True
    )
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    return {
        "area": area,
        "perimeter": perimeter,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "hull_area": hull_area,
        "solidity": solidity
    }

def build_dataset():
    img_features = []
    for i in os.scandir("img/data"):
        if i.is_dir():
            label = i.name
            for j in os.scandir(i.path):
                if j.is_file():
                    image = cv2.imread(j.path)
                    
                    if image is None:
                        raise ValueError("Error opening " + j.path + " image.")
                    gray = get_gray_scale_and_histogram(image)
                    mask = get_mask_from_grayscale(gray)
                    contour = get_contour_from_mask(image, mask)
                    features = extract_basic_features(contour)
                    features["label"] = label
                    img_features.append(features)
    save_to_csv(img_features)


def save_to_csv(features):

    # Sample NumPy array of dictionaries
    data = np.array(features)

    # 1. Extract column names for the header
    headers = ",".join(data[0].keys())

    # 2. Extract values into a 2D NumPy array
    rows = np.array([list(d.values()) for d in data])

    # 3. Save to CSV (fmt='%s' handles mixed types; comments='' removes the default '#' prefix)
    np.savetxt("output/dataset.csv", rows, delimiter=",", header=headers, comments="", fmt="%s")
