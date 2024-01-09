import os
import util.utils as utils
import models.model as model
from models.custom_metrics import psnr_metric, ssim_metric
from models.custom_loss import simplified_perceptual_loss
import tensorflow as tf
from PIL import Image
import numpy as np

def load_model(model_path):
    custom_objects = {"psnr_metric": psnr_metric,
                      "ssim_metric": ssim_metric,
                      "simplified_perceptual_loss": simplified_perceptual_loss
                      }
    return tf.keras.models.load_model(model_path,custom_objects=custom_objects)

def main():
    model_path = os.path.join(os.getcwd(), 'models', 'modelesrcnn8.h5')
    model = load_model(model_path)
    
    image_path = input("Please enter the path to your PNG image: ")
    try:
        original_image = Image.open(image_path)
        cropped_image = utils.crop_to_largest_64x64(original_image)
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        cropped_image.save(os.path.join(output_folder,"cropped_LR_img.png"))
        patches = utils.divide_into_patches(cropped_image)
        mean_predictions, uncertainty_maps, raw_predictions = utils.model_predict(model, patches)
 
        num_cols = cropped_image.width // 64
        num_rows = cropped_image.height // 64
        stitched_one_image = utils.stitch_patches(raw_predictions, num_cols, num_rows)
        stitched_one_image.save(os.path.join(output_folder,"one_prediction.png"))

        stitched_mean_image = utils.stitch_patches(mean_predictions, num_cols, num_rows)
        stitched_mean_image.save(os.path.join(output_folder,"mean_prediction.png"))

        heatmap_images = [utils.generate_heatmap_image(uncertainty_patch) for uncertainty_patch in uncertainty_maps]
        stitched_uncertainty_map = utils.stitch_patches(heatmap_images, num_cols, num_rows, is_heatmap=False)
        stitched_uncertainty_map.save(os.path.join(output_folder,"uncertainty_map.png"))
        
        uncertainty_scores = [np.mean(uncertainty) for uncertainty in uncertainty_maps]
        num_images = len(uncertainty_scores)
        num_to_select = min(num_images, 5)  # up to 5 images, or fewer if not enough images

        most_certain_indices = np.argsort(uncertainty_scores)[:num_to_select]
        least_certain_indices = np.argsort(uncertainty_scores)[-num_to_select:]
        
        mean_pred_images = [Image.fromarray((patch * 255).astype(np.uint8)) for patch in mean_predictions]

        utils.save_patches("output/details/most_certain", most_certain_indices, patches, mean_pred_images, heatmap_images)
        utils.save_patches("output/details/least_certain", least_certain_indices, patches, mean_pred_images, heatmap_images)
        utils.save_colorbar("output/colorbar.png")
        
        print(f"Pre-enhanced, cropped image saved at  {os.path.join(output_folder,'cropped_LR_img.png')}")
        print(f"One predicted image saved at {os.path.join(output_folder,'one_prediction.png')}")
        print(f"Mean predicted image saved at {os.path.join(output_folder,'mean_prediction.png')}")
        print(f"Heat map saved at {os.path.join(output_folder,'uncertainty_map.png')}")
            

    except IOError:
        print("Error in processing the image.")

if __name__ == "__main__":
    main()
