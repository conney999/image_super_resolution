# Super-Resolution ML Project

## Overview

This project focuses on super-resolution, a process that enhances the quality of lower-resolution images using the Enhanced Super-Resolution Convolutional Neural Network (ESRCNN) deep learning technique. This version of ESRCNN includes two dropout layers for uncertainty quantification. The main script (`main.py`) serves as the entry point for users, allowing them to provide an image via the command line. The model then generates an enhanced resolution output, along with an uncertainty map and other intermediate outputs.

## Project Structure

- **Data**: Contains datasets used for training and validating the model, including high-resolution (HR) and low-resolution (LR) image pairs.
- **Images**: Stores images relevant to the project, such as sample outputs.
- **Models**: Includes saved model files and the `model.py` script, which defines the trained super-resolution model architecture.
- **Notebooks**: Jupyter notebooks for exploratory data analysis, model training, evaluation, and other experiments.
- **Util**: Utility scripts for image processing and model operations.

## Main Components

- **main.py**: The application's entry point. It loads a trained model, processes a user-inputted image, and performs super-resolution. It also generates uncertainty maps and various intermediate outputs.
- **image_utils.py**: Functions for image preprocessing, augmentation, and displaying sample images.
- **utils.py**: Utility functions for image processing, model predictions, and generating heatmaps.
- **custom_loss.py**: Defines a custom loss function (`simplified_perceptual_loss`) used in the model training.
- **custom_metrics.py**: Custom metrics like PSNR and SSIM for model evaluation.
- **model.py**: Defines the ESRCNN model architecture used for super-resolution.

## Setup and Execution

1. **Environment Setup**:
   - Ensure the proper environment and dependencies are set up.
   - Run `conda env create -f environment.yml` to create the environment.

2. **Running the Main Script**:
   - Activate the environment with `conda activate sr-env`.
   - Run `main.py` and provide the absolute path to a PNG image when prompted.
   - The script processes the image, performs super-resolution, and saves outputs in the `output` directory.

3. **Understanding Outputs**:
   - Various images are saved, including the cropped input, predictions, uncertainty maps, and details of certain patches.

4. **Model Exploration**:
   - Use the notebooks in the `Notebooks` directory for a step-by-step exploration of model training techniques and outputs.
