from pathlib import Path

import cv2
import joblib

from bakery.process import get_contour_from_mask, get_gray_scale_and_histogram, get_mask_from_grayscale

from .train import train_model
from .extract import build_dataset, extract_basic_features
import argparse
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Bread slice tester")
    subparsers = parser.add_subparsers(dest="command", required=True)
    test_parser = subparsers.add_parser("test", help="Run tests on a specific path")
    test_parser.add_argument("path", type=Path, help="Path to the test file or directory")

    args = parser.parse_args()
    if args.command == "build":
        build_dataset()
        train_model()

    elif args.command == "test":
        image = cv2.imread(args.path)
        
        if image is None:
            raise ValueError("Error opening " + args.path + " image.")
        
        model_path = "output/bakery/model.pkl"
        model = joblib.load(model_path)
         
        if model is None:
            raise ValueError("Error opening " + model_path + " image.")
        
        gray = get_gray_scale_and_histogram(image)
        mask = get_mask_from_grayscale(gray)
        contour = get_contour_from_mask(image, mask)
        features = extract_basic_features(contour)

        probabilities = model.predict_proba(pd.DataFrame([features]))

        for label, probability in zip(model.classes_, probabilities[0]):
            print(f"{label}: {probability * 100:.2f}%")
    
