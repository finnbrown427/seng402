# seng402 - Finding Small Things in Big Images

## Description
Pipeline for generating and pre-processing a dataset of small targets hidden in noise, then training a CNN to detect these targets. After training a "heat map" visualisation of where target are predicted to be is produced.

## Setup Instructions
#### **1) Create & activate virtual enviroment**
In your local directory:<br>  
python3 -m venv .venv  
source .venv/bin/activate

#### **2) Upgrade pip**
python -m pip install --upgrade pip

#### **3) Install dependencies**
python -m pip install -r requirements.txt

If the environment gets into a bad state, delete `.venv/` and run these steps again.

## Usage
**To train target detection and generate visulisation image**<br>  
python train_iterable.py

## Recreating The Environment

This project is set up to be recreated from scratch with the same steps every time:

1. Remove any existing `.venv/` folder.
2. Create a new virtual environment with `python3 -m venv .venv`.
3. Activate it and run `python -m pip install --upgrade pip`.
4. Install packages with `python -m pip install -r requirements.txt`.

If Torch still fails to import, try using Python 3.12 or 3.11 for the virtual environment, since Torch wheels can be sensitive to the Python version.

## Project Structure

- image_generation.py: Synthetic image generation, target embedding and image segmentation
- pytorch_cnn.py CNN model definition
- train_detection.py Legacy training loop using one random segment per image
- train_iterable.py Iterable dataset training using all segments per image. Also produces a heatmap visualisation of detections

## Author
**Finn Brown**