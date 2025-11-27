# 🫁 Deep Learning-based Diagnostic System for Chest X-ray Image Classification

## Introduction

This project develops a **Deep Learning-based Diagnostic System** designed for the automatic classification of chest X-ray images. The goal is to assist medical professionals in rapidly and accurately screening for common pulmonary conditions (e.g., Pneumonia, Atelectasis, Cardiomegaly,...).

## Key Features

* **Multi-Class Classification:** Classifies X-ray images into several distinct pathological classes (e.g., Atelectasis, Pneumonia,...).
* **Diagnostic Support:** Provides probability scores for each diagnosis, helping physicians make quicker decisions.
* **Automated Preprocessing:** Integrates an image preprocessing pipeline (`DataPreprocessing.ipynb`) to standardize input data.
* **Demo Interface:** Provides a simple interface (`Demo.py`) for users to upload X-ray images and receive instant diagnostic results.

## Technology Stack

The project is built on the Python platform, relying on the following core libraries:

* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch
* **Image/Data Processing:** NumPy, Pandas, OpenCV, PIL
* **Interface:** Tkinter

## Installation and Setup

Follow these steps to set up the environment and run the system:

### 1. Clone the Repository

```bash
git clone [https://github.com/PeterMon0905/Diagnostic-System-for-Chest-X-ray-Image-Classification.git](https://github.com/PeterMon0905/Diagnostic-System-for-Chest-X-ray-Image-Classification.git)
cd Diagnostic-System-for-Chest-X-ray-Image-Classification
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Download Weight
Download this weight file and save it to the project root folder.
[model.pth](https://github.com/PeterMon0905/Diagnostic-System-for-Chest-X-ray-Image-Classification/releases/download/v1.0.0/model.pth)
## Usage
Running the Diagnostic Demo
```bash
python Demo.py
```
