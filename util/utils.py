import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import io
import os

####  Image processing functions
def crop_to_largest_64x64(img):
    '''
        Function to center crop image to the largest possible center areaa that consists of 64x64 pixel patches
        Returns a PIL Image object
    '''
    width, height = img.size
    print(f"The image you inputted as width: {width}, height:{height}.")
    if width < 64 or height < 64:
        print("The image you inputted has invalid dimensions. Please consult READme.")
        return None
    else:
        # Calculate cropping dimensions
        crop_width = 64 * (width // 64)
        crop_height = 64 * (height // 64)
        
        print(f"It's being center cropped to width: {crop_width}, height:{crop_height}.")
        
        # Calculate the top-left point for the crop to be centered
        left = (width - crop_width) / 2
        top = (height - crop_height) / 2
        
        # Calculate the bottom-right point of the crop
        right = (width + crop_width) / 2
        bottom = (height + crop_height) / 2

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img
        

def divide_into_patches(image, patch_size=64):
    '''
        Function that divides up an image of dimensions that are multiples of 64
        Returns a list of PIL image objects
    '''
    width, height = image.size
    patches = []
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            patch = image.crop((i, j, i + patch_size, j + patch_size))
            patches.append(patch)
        
    return patches
   

def normalize_and_convert_to_image(patch):
    # Normalize to 0-1
    patch_normalized = (patch - patch.min()) / (patch.max() - patch.min())
    # Scale to 0-255 and convert to uint8
    patch_scaled = (patch_normalized * 255).astype('uint8')
    return Image.fromarray(patch_scaled, 'RGB')
    

def stitch_patches(patches, num_cols, num_rows, patch_size=128, is_heatmap=False):
    stitched_image = Image.new('RGB', (num_cols * patch_size, num_rows * patch_size))
    
    for i, patch in enumerate(patches):
        if isinstance(patch, np.ndarray):
            patch_image = normalize_and_convert_to_image(patch)
        else:
            patch_image = patch
        col = i // num_rows
        row = i % num_rows
        stitched_image.paste(patch_image, (col * patch_size, row * patch_size))

    return stitched_image

   

### Model and uncertainty quantification related functions

def monte_carlo_prediction(model, patch_tensor, num_passes=40):
    predictions = [model(patch_tensor, training=True) for _ in range(num_passes)]
    return np.array(predictions)
    
    
def model_predict(model, patches):
    mean_predictions = []
    uncertainty_maps = []
    raw_predictions = []

    for patch in patches:
        patch_np = np.array(patch) / 255.0
        patch_tensor = tf.convert_to_tensor(patch_np, dtype=tf.float32)
        patch_tensor = tf.expand_dims(patch_tensor, axis=0)  # Add batch dimension

        mc_predictions = monte_carlo_prediction(model, patch_tensor)
        mean_prediction = np.mean(mc_predictions, axis=0).squeeze(0)
        uncertainty = np.std(mc_predictions, axis=0).squeeze(0)
        raw_prediction = mc_predictions[0].squeeze(0)

        mean_predictions.append(mean_prediction)
        uncertainty_maps.append(uncertainty)
        raw_predictions.append(raw_prediction)  # use first prediction as one enhancement

    return mean_predictions, uncertainty_maps, raw_predictions
    
    
def generate_heatmap_image(uncertainty_patch, vmax=0.1):
    plt.figure(figsize=(6, 6))
    plt.imshow(uncertainty_patch.mean(axis=2), cmap='cool', interpolation='nearest', vmin=0, vmax=vmax)
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)  # Set higher DPI
    plt.close()
    buf.seek(0)
    heatmap_img = Image.open(buf)
    heatmap_img = heatmap_img.resize((128, 128)) #resize to match model output

    return heatmap_img


def save_patches(folder_name, indices, lr_patches, mean_predictions, heatmap_images):
    os.makedirs(folder_name, exist_ok=True)
    for i, idx in enumerate(indices):
        lr_patches[idx].save(os.path.join(folder_name, f"original_lr_patch_{i}.png"))
        mean_predictions[idx].save(os.path.join(folder_name, f"mean_pred_{i}.png"))
        heatmap_images[idx].save(os.path.join(folder_name, f"heatmap_{i}.png"))


def save_colorbar(output_path, cmap='cool', vmin=0, vmax=0.1):
    fig, ax = plt.subplots(figsize=(2, 6))
    # Create a color map scalar mappable object
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax)
    cbar.set_label('Uncertainty', rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
