import cv2
import argparse
import sys
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, 'cropped.jpg')

img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read the image from the path: {image_path}")
else:
    text = pytesseract.image_to_string(img)
    print(text)