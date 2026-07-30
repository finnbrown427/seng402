# seng402 - Finding Small Things in Big Images

## Description
Pipeline for generating and pre-processing a dataset of small targets hidden in noise, then training a CNN to detect these targets. After training a "heat map" visualisation of where target are predicted to be is produced.

## Setup Instructions
#### **1) Create & activate virtual enviroment**
In your local directory:<br>  
python3 -m venv venv  
source venv/bin/activate

#### **2) Install dependencies**
python -m pip install numpy opencv-python torch torchvision torchaudio

## Usage
**To train target detection and generate visulisation image**<br>  
python train_iterable.py

## Project Structure

- image_generation.py: Synthetic image generation, target embedding and image segmentation
- pytorch_cnn.py CNN model definition
- train_detection.py Legacy training loop using one random segment per image
- train_iterable.py Iterable dataset training using all segments per image. Also produces a heatmap visualisation of detections

## Author
**Finn Brown**
