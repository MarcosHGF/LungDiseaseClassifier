# Lung Disease Classification using Deep Learning

This project is a classification system for detecting lung diseases, including **pneumonia** and **tuberculosis**, using **X-ray images**. It employs a **Convolutional Neural Network (CNN)** to categorize X-ray images into different disease categories.

## Dataset Used

- https://www.kaggle.com/datasets/rifatulmajumder23/combined-unknown-pneumonia-and-tuberculosis

## Technologies Used

- **Flask**: Python web framework for creating the user interface.
- **Pytorch**: For building and training the CNN model.
- **PIL**: Python Imaging Library for image handling.
- **NumPy**: For array manipulations.
- **Requests**: To handle file uploads.

## Features

- **Image Upload**: Upload X-ray images to classify them into disease categories.
- **Prediction Output**: The model predicts whether the image belongs to normal lungs, pneumonia, tuberculosis, or non-X-ray.

## How it Works

1. **Data Collection**: Collects X-ray images for training and testing.
2. **Data Preprocessing**: Images are resized and normalized before feeding into the model.
3. **Model**: A CNN model is used to classify X-ray images.
4. **Web Interface**: A simple interface built with Flask allows users to upload images for classification.

## How to Run the Project

### Requirements

Ensure Python 3.x is installed and install the required dependencies:

```bash
pip install flask torch torchvision numpy kagglehub
