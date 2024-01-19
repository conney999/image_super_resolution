# Super-Resolution ML Project

## Overview

This project focuses on super-resolution, a process that enhances the quality of lower-resolution images using the Enhanced Super-Resolution Convolutional Neural Network (ESRCNN).It specifically targets images downscaled by a factor of two using bicubic interpolation, enhancing 64x64 pixel patches into higher quality 128x128 pixel patches. This version of ESRCNN includes two dropout layers for uncertainty quantification.

## User Interaction:
The main script (`main.py`) serves as the entry point for users, allowing them to provide an image via the command line. When you provide an image as input to our project, the following steps are performed:

1. **Center Crop:**
   - Begin by center cropping your image to create patches of the largest possible size that are each 64x64 pixels. This makes each image suitable as model input
2. **Enhancement with Uncertainty:**
   - For each of these 64x64 cropped image patches, the trained model generates a higher quality 128x128 image patch.
   - The model is run for a total of 40 times for each image patch. This technique known as Monte Carlo Dropout
   - This repetition allows us to capture uncertainty in the predictions. Think of it as the model making slightly different predictions each time, which helps us understand its level of confidence.

3. **User Outputs:**
   - When you receive the results, it will be in the 'output' directory, key outputs
      - **Cropped_LR_Image:** This is the cropped portion of your input image that you started with, without any enhancement.
      - **One_Prediction:** One prediction that represents the result of result of the model after 1 pass.
      - **Mean_Prediction:** One prediction that represents the result of the model after averaging the 40 different predictions we obtained during the uncertainty analysis.
      - **Uncertainty_map:** To give you insights into how confident the model is at different parts of the image, there is a heatmap. 
      - **Colorbar:** The scale for the heatmap
      - **Details Folder:** You'll receive a "details" folder that contains two subfolders: "most certain" and "least certain."
         - **Most Certain:** This subfolder includes patches for which the model is most certain about its predictions.
         - **Least Certain:** This subfolder includes patches for which the model is least certain about its predictions.
         -  **File Naming Convention:** Inside least certain and most certain, you will find a series of corresponding triplets, all in PNG format, and the'i' index corresponds to the triplets and varies accordingly :
            - **Original LR Patch** (e.g., original_lr_patch_i.png): These will be the lower-resolution patches that you used as input to the model.
            - **Mean Predicted Patch** (e.g., mean_pred_patch_i.png): These will be the mean enhanced patches generated by the model after 40 passes. 
            - **Heatmap** (e.g., heatmap_i.png): These will be the heatmaps indicating the confidence level of the model's predictions for each patch.


4. **Patch Stitching:**
   - It's important to note that there are no additional post-processing or blending of the enhanced patches. Instead, the enhanced patches are simply stitched back together in the order they were cropped.
   
    - By following this approach, the model reesults can properly be analyzed while preserving the image's details and being transparent about how the model worked and the confidence in its predictions.


## Project Structure

- **Data**: Contains datasets used for training and validating the model, including high-resolution (HR) and low-resolution (LR) image pairs.
- **Images**: Images I supply that you can input to the model if you wish
- **Models**: Includes saved model file, the `model.py` script, which defines the trained super-resolution model architecture, and other custom objects  
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
   - Run `python3 main.py` and provide the absolute path to a PNG image when prompted.
   - I have included some sample images in the Images folder you can use, just provide the path to that image when prompted.
    - If you decide to use your own image, make sure it is low-resolution and downscaled by a factor of 2 via bicubic interpolation. I provide details on where you can download some of these in the Datasets section
   - The script processes the image, performs super-resolution, and saves outputs in the `output` directory.


3. **Understanding Outputs**:
   - Various images are saved, including the cropped input, prediction,mean prediction, uncertainty map,color bar for the heatmap, and details of certain patches, look under User Interaction,User Outputs above for more detail

4. **Model Exploration**:
   - Can use the notebooks in the `Notebooks` directory for a step-by-step exploration of model training techniques and outputs.

## Datasets
My model was trained on DIV2K images 800-900 in their validation series. Feel free to use your own images as long as they are low-resolution and downscaled by a factor of 2 using bicubic interpolation. Some images you can easily download is from DIV2K(besides 800-900) Flickr2K, BSD100, Set14. Make sure you get the images ending in 'x2_LR.png' or  '_SRF_2.png' and under bicubic/X2 or image_SRF_2/ subfolders to ensure you get the correct low-resolution image. 

**Dataset Preparation**:
The dataset I used to train originated from 70 image pairs from the DIV2K collection, each consisting of a low-resolution (LR) image downsampled by a factor of two using bicubic interpolation and its corresponding high-resolution (HR) counterpart. After splitting these pairs into training, validation, and test sets, the training set comprised 42 whole image pairs. For each LR-HR image pair, 10 random patch pairs were extracted.Each pair was a 64x64 pixel LR patch and its corresponding 128x128 pixel HR patch, resulting in a total of 420 LR-HR patch pairs for model training. I used cropped image pairs instead of whole image pairs to promote a richer variety of textures and patterns in the training data.
