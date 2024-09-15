# YOLOv8 Object Detection with FastAPI

This project demonstrates how to prepare a dataset, train a YOLOv8 model, and create a FastAPI backend for object detection.

## Project Overview

1. **Dataset Preparation**
2. **Model Training**
3. **FastAPI Backend Development**

## 1. Dataset Preparation

### XML to YOLO Format

- **Script**: `xml_to_txt.py`
- **Description**: Converts XML annotations from the Kaggle dataset ([PCB Defects Dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects/data)) into the text format required by YOLOv8.
- Download the dataset inside the Project folder and unzip folder ( name directory as PCB_Dataset).
- **Command Used**:
  ```bash
    python xml_to_txt.py

- **Output** : It will generate /Input_Data folder with all labels .txt files for YOLO input.

### Data Splitting

- **Script**: `data.py`
- **Description**: Splits the dataset into training, validation, and test sets in YOLO format.
- **Command Used**:
  ```bash
    python data.py
- **Output** : It will generate /dataset folder with 2 subfolders images and label
    ```
    dataset/
    │
    ├── images/
    │   ├── train/    # Training images
    │   ├── val/      # Validation images
    │   └── test/     # Test images
    │
    ├── labels/
    │   ├── train/    # YOLO format labels for training images
    │   ├── val/      # YOLO format labels for validation images
    │   └── test/     # YOLO format labels for test images
    ```


### Dataset YAML Configuration

- **File**: `dataset.yaml`
- **Description**: Specifies the directory structure and class names for YOLOv8.

## 2. Model Training

### Training Command

- **Command Used**:
  ```bash
  yolo train model=yolov8n.yaml data=dataset.yaml epochs=5 batch=32 imgsz=640
  
- model=yolov8n.yaml: Configuration file for YOLOv8 model.
- data=dataset.yaml: Path to the dataset YAML file.
- epochs=5: Number of training epochs.
- batch=32: Batch size.
- imgsz=640: Image size.

## 3. FastAPI Backend Development

### FastAPI Script

- **Script**: `FastAPI.py`
- **Description**: Implements the backend API with the following endpoints:
  - `/predict (POST)`: Accepts image data and confidence limit, and returns predictions (bounding boxes and labels) as structured data.
  - `/visualize (POST)`: Accepts image data and confidence limit, and returns the image with drawn bounding boxes and label annotations.

### Running FastAPI

1. **Install Dependencies**:
   - Install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Start the FastAPI Server**:
   ```bash
   uvicorn FastAPI:app --reload
