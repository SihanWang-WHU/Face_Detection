# Face Detection
Two ways of detecting faces from a video stream (dlib + Mask R-CNN Model)

## 1. dlib
This script employs the `cv2` and `face_recognition` libraries to detect and crop faces from a given video. The function `detect_and_crop_faces` takes a video file path as input. 
It opens the video file using OpenCV (`cv2.VideoCapture`). 
The script iterates through each frame of the video, using the `face_recognition` library to identify face locations within these frames. 
For each detected face, it crops the image around the face, optionally resizes it to a standard size (e.g., 224x224 for compatibility with models like ResNet), converts it to an array, and stores these face arrays in a list. 
Finally, the script returns this list of cropped face images.

## 2. Mask R-CNN
### 2.1 WIDER Face Dataset Introduction

The WIDER Face dataset is a comprehensive benchmark for face detection tasks, 
known for its diversity in terms of scale, pose, and occlusion. This makes it one 
of the most challenging and widely-used datasets in the field of face detection. 
The dataset consists of thousands of images sourced from the internet, 
encompassing a variety of scenarios including social events, public spaces, 
and daily activities. Each image in the dataset comes with annotations for face 
bounding boxes, providing essential data for training and evaluating face 
detection models.

In this implementation, a custom PyTorch dataset class, `WIDERFaceDataset`, 
is created to efficiently handle the specific structure and annotation format 
of the WIDER Face dataset. Key features of this class include:

- **Reading and Parsing Annotation Files**: The class is equipped to read 
  annotation text files, extracting image paths and bounding box labels. 
  These annotations are crucial for training face detection models, 
  offering ground truth for the locations of faces in the images.

- **Data Preprocessing and Transformations**: Methods are included for preprocessing 
  images and applying various transformations. These are essential for data 
  augmentation and enhancing the model's robustness to variations in input images.

- **Custom Collate Function**: The `my_collate_fn` function is utilized in the data 
  loader for efficient batching of images and their corresponding target annotations, 
  which is vital for training models using mini-batch gradient descent.

- **Visualization Tool**: A visualization function, `visualize_sample`, is also included 
  to display images with their annotated bounding boxes. This tool aids in debugging 
  and provides a clearer understanding of the dataset.

### 2.2 Model Introduction

In designing this model for face detection, I chose the **Mask R-CNN architecture** with a **ResNet-50 backbone** and **Feature Pyramid Network (FPN)** for their distinct advantages.

- **Mask R-CNN Framework**: Renowned for object detection and instance segmentation, Mask R-CNN is ideal for accurately identifying and segmenting faces. Its unique ability to add a segmentation branch to the standard object detection framework enhances precision in demarcating facial boundaries.

- **ResNet-50 Backbone**: Integrated for its deep residual learning capability, crucial in capturing detailed facial features. This depth is vital for recognizing a wide range of facial characteristics under various conditions.

- **Feature Pyramid Network (FPN)**: Added as a strategic decision to handle faces at different scales. FPN's multi-scale feature extraction is essential in detecting faces of various sizes, a common challenge in real-world scenarios.

For the **classifier and mask predictor heads**, I tailored them specifically for face detection:
- The **classifier head** distinguishes faces from the background.
- The **mask predictor** accurately segments faces, which is critical for applications requiring detailed facial analysis.

### 2.3 Model Training Process

To train my face detection model, I implemented a structured approach:

1. **Data Processing and Transformation**: Utilized `WIDERFaceDataset` for data processing and applied transformations including resizing, tensor conversion, and normalization.

2. **Data Loader Setup**: Configured data loaders for both training and validation datasets with custom collation function `my_collate_fn`.

3. **Model Initialization**: Employed `get_model_instance_segmentation` to create the Mask R-CNN model with the ResNet-50 backbone and initialized it on the appropriate device (GPU or CPU).

4. **Optimizer and Scheduler**: Selected SGD optimizer with specific learning rate, momentum, and weight decay settings. Incorporated a learning rate scheduler for adaptive learning rate adjustments.

5. **Training Loop**: Executed a training loop for a predetermined number of epochs, integrating `train_one_epoch` function for training and learning rate scheduling.

6. **Model Saving**: After training, saved the model's state for future use.

This process ensured efficient and effective training of the model, enabling it to learn robust features for accurate face detection.

