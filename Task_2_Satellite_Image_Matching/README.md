# Task 2: Sentinel-2 Satellite Image Matching

This repository contains the solution for Task 2 of the Data Science Test Task: matching Sentinel-2 satellite images of the same area taken in different seasons.


## Requirements

To install all necessary dependencies, navigate to the project root directory and run:
./Data_Science_Test_Task/Task_2_Satellite_Image_Matching/

```pip install -r requirements.txt```

## Usage

### **Step 1: Prepare the Dataset**
Follow the instructions for dataset preparation in the Jupyter Notebook: [dataset_preparation.ipynb](../Task_2_Satellite_Image_Matching/notebooks/dataset_preparation.ipynb).

### **Step 2: Train the Model**
Train the feature matching model using the sorted dataset:

   ```python3 Task_2_Satellite_Image_Matching/src/train_model.py```


### **Step 3: Perform Inference
Run inference on random pairs of images from the test_images/ folder:

```python3 Task_2_Satellite_Image_Matching/src/infer_model.py```


Key Notes
Images are sorted by tile (location), and only True Color Images (_TCI.jp2) are used.
The model matches features between two images of the same area taken in different seasons.
Results are visualized using matplotlib.

