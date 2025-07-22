# Automatic Number Plate Recognition (ANPR)

![Example License Plate](https://raw.githubusercontent.com/sameerpashaaa/ANPR/master/implementation/exampleimg.jpg)


This project implements an Automatic Number Plate Recognition (ANPR) system using Python, OpenCV, and Tesseract OCR. The system is designed to detect license plates in an image, crop the plate region, and then recognize the alphanumeric characters on the plate.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [1. License Plate Detection](#1-license-plate-detection)
  - [2. Image Preprocessing](#2-image-preprocessing)
  - [3. Character Recognition (OCR)](#3-character-recognition-ocr)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

-   **License Plate Detection**: Utilizes a pre-trained YOLOv3 model to accurately detect and localize license plates in images.
-   **Character Recognition**: Employs Tesseract OCR to extract the license plate number from the cropped plate image.
-   **Image Preprocessing**: Includes steps for resizing, grayscaling, noise reduction, and binarization to improve OCR accuracy.
-   **Customizable**: Easy to configure paths for model weights, configuration files, and class names.

## Project Structure

```
Anppr/
└── ANPR/
    ├── darknet/                # YOLO configuration and data
    │   ├── cfg/yolov3.cfg
    │   └── data/obj.names
    ├── weights/                # Model weights
    │   └── final_weights/yolov3_final.weights
    ├── samples/                # Sample images for testing
    ├── main.py                 # Main script for detection and recognition
    └── README.md               # This file
```

## Prerequisites

Before running the project, you need to have the following installed:

-   Python 3.x
-   OpenCV for Python
-   Pytesseract
-   NumPy
-   Tesseract OCR Engine

## Installation

1.  **Clone the repository:**

    ```bash
    git clone < https://github.com/sameerpashaaa/ANPR.git>
    cd ANPR
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install opencv-python pytesseract numpy
    ```

3.  **Install Tesseract OCR:**

    -   **Windows**: Download and install Tesseract from the [official Tesseract repository](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to add the Tesseract installation directory to your system's PATH or specify the path in the `main.py` script:
        ```python
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ```
    -   **macOS**: `brew install tesseract`
    -   **Linux**: `sudo apt-get install tesseract-ocr`

4.  **Download YOLOv3 Weights:**

    Download the pre-trained `yolov3_final.weights` file and place it in the `weights/final_weights/` directory.

## Usage

To run the ANPR system, execute the `main.py` script from your terminal:

```bash
python main.py
```

The script will then prompt you to enter the path to the image you want to process:

```
Enter the path to the image: path/to/your/image.jpg
```

The system will process the image and print the detected license plate number to the console.

## How It Works

The process is divided into three main stages:

### 1. License Plate Detection

The `main.py` script loads a pre-trained YOLOv3 model to detect license plates in the input image. The model identifies the bounding box coordinates of the plate.

### 2. Image Preprocessing

Once the license plate is cropped from the main image, it undergoes several preprocessing steps to enhance its quality for OCR:

-   **Resizing**: The image is enlarged to make the characters clearer.
-   **Grayscale Conversion**: The image is converted to grayscale.
-   **Noise Reduction**: A median blur is applied to remove noise.
-   **Binarization**: A threshold is applied to create a black-and-white image, which is ideal for OCR.

### 3. Character Recognition (OCR)

The preprocessed image is passed to the Tesseract OCR engine, which analyzes the image and extracts the alphanumeric characters, returning the recognized license plate number.

## Configuration

-   **Model Paths**: The paths to the YOLOv3 configuration file, weights, and class names can be modified in the `Args` class within `main.py`.
-   **Tesseract Path**: The path to the Tesseract executable can be set at the beginning of `main.py`.
-   **OCR Parameters**: The Tesseract configuration, including Page Segmentation Mode (PSM) and character whitelisting, can be adjusted in the `recognize_plate` function to optimize performance.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```
        
