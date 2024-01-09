
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import random
import tensorflow as tf

def make_subset(hr_source_dir, lr_source_dir, dest_dir, sample_size, start_index=-1):
  if not os.path.exists(hr_source_dir) or not os.path.exists(lr_source_dir):
    print("Source directories not found")
    return

  hr_filenames = sorted(os.listdir(hr_source_dir))
  random.seed(42)
  sample_high_res_filenames = random.sample(hr_filenames, sample_size)

  hr_dest_dir = os.path.join(dest_dir, "HR")
  os.makedirs(hr_dest_dir, exist_ok=True)

  lr_dest_dir = os.path.join(dest_dir, "LR")
  os.makedirs(lr_dest_dir, exist_ok=True)

  for hr_fname in sample_high_res_filenames:
    if start_index > -1:
      lr_fname = hr_fname.replace("HR.png", "LR.png")
      shutil.copy(os.path.join(hr_source_dir, hr_fname),
                os.path.join(hr_dest_dir, hr_fname))
      os.rename(os.path.join(hr_dest_dir, hr_fname),
                os.path.join(hr_dest_dir, f"{start_index+1}_HR.png"))

      shutil.copy(os.path.join(lr_source_dir, lr_fname),
                os.path.join(lr_dest_dir, lr_fname))
      os.rename(os.path.join(lr_dest_dir, lr_fname),
                os.path.join(lr_dest_dir, f"{start_index+1}_LR.png"))
      start_index+=1
    else:
      shutil.copy(os.path.join(hr_source_dir, hr_fname),
                os.path.join(hr_dest_dir, hr_fname))
      lr_fname = hr_fname.replace(".png", "x2.png")
      shutil.copy(os.path.join(lr_source_dir, lr_fname),
                os.path.join(lr_dest_dir, lr_fname))

def get_min_image_dim(folder_path):
    sizes = {}
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        with Image.open(img_path) as img:
            sizes[img_file] = img.size  # img.size is a tuple (width, height)

    min_width = 2**31
    min_height = 2**31
    for image_name, size in sizes.items():
     # print(f"{image_name}: Width = {size[0]}, Height = {size[1]}")
       if size[0] < min_width: min_width = size[0]
       if size[1] < min_height: min_height = size[1]
    print(f"Min width, height: {min_width, min_height}")

def display_sample_images(file_list, title,is_lr=True):
    plt.figure(figsize=(10, 5))
    dir = '/content/gdrive/MyDrive/super-res_project/Data/raw/raw_DIV2K_subset'
    if is_lr: res_type = "/LR/"
    else: res_type = "/HR/"
    for i, file in enumerate(file_list[:2]):
        image = Image.open(dir+res_type+file)
        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    plt.show()

def augment_patch(lr_patch, hr_patch):
    """
    Function to process a pair of HR and LR images: randomly rotate an aligned patch pair
    """
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        lr_patch = tf.image.flip_left_right(lr_patch)
        hr_patch = tf.image.flip_left_right(hr_patch)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        lr_patch = tf.image.flip_up_down(lr_patch)
        hr_patch = tf.image.flip_up_down(hr_patch)

    # Random rotation
    rotation_choice = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    lr_patch = tf.image.rot90(lr_patch, k=rotation_choice)
    hr_patch = tf.image.rot90(hr_patch, k=rotation_choice)

    return lr_patch, hr_patch

def center_crop_images(lr_image, hr_image, patch_size=64, scale=2):
    """
    Function to process a pair of HR and LR images: center crop an aligned patch pair
    """
    # Get LR and HR images size
    hr_height, hr_width = tf.shape(hr_image)[0], tf.shape(hr_image)[1]
    lr_height, lr_width = tf.shape(lr_image)[0], tf.shape(lr_image)[1]

    # Calculate the top-left corner of the center patch for HR image
    hr_y_center = hr_height // 2
    hr_x_center = hr_width // 2
    hr_y = hr_y_center - (patch_size * scale) // 2
    hr_x = hr_x_center - (patch_size * scale) // 2

    # Ensure the crop dimensions do not exceed the image dimensions
    hr_y = tf.clip_by_value(hr_y, 0, hr_height - patch_size * scale)
    hr_x = tf.clip_by_value(hr_x, 0, hr_width - patch_size * scale)

    # Extract HR center patch
    hr_patch = tf.image.crop_to_bounding_box(hr_image, hr_y, hr_x, patch_size * scale, patch_size * scale)

    # Calculate the top-left corner of the center patch for LR image
    lr_y = hr_y // scale
    lr_x = hr_x // scale

    # Extract LR center patch
    lr_patch = tf.image.crop_to_bounding_box(lr_image, lr_y, lr_x, patch_size, patch_size)

    return lr_patch, hr_patch

def random_crop_images(lr_image, hr_image, patch_size=64, scale=2):
    """
    Function to process a pair of HR and LR images: randomly crop an aligned patch pair
    """
    # Get LR and HR images size
    hr_height, hr_width = tf.shape(hr_image)[0], tf.shape(hr_image)[1]
    lr_height, lr_width = tf.shape(lr_image)[0], tf.shape(lr_image)[1]

    # # Adjust HR dimensions for cropping dimensions
    hr_height = hr_height - tf.math.mod(hr_height, scale)
    hr_width = hr_width - tf.math.mod(hr_width, scale)

    # Randomly select a top-left corner of the HR patch
    hr_y = tf.random.uniform(shape=(), maxval=hr_height - patch_size * scale, dtype=tf.int32)
    hr_x = tf.random.uniform(shape=(), maxval=hr_width - patch_size * scale, dtype=tf.int32)

    # Extract HR patch
    hr_patch = tf.image.crop_to_bounding_box(hr_image, hr_y, hr_x, patch_size * scale, patch_size * scale)

    # Extract corresponding LR patch
    lr_y = hr_y // scale
    lr_x = hr_x // scale
    lr_patch = tf.image.crop_to_bounding_box(lr_image, lr_y, lr_x, patch_size, patch_size)

    return lr_patch, hr_patch

def process_image(lr_path, hr_path, num_patches, patch_size=64, scale=2, augment=False, apply_func=random_crop_images, *args, **kwargs):
    """
    Function to process a pair of HR and LR images: Load, extract, augment multiple aligned patches.
    """
    # Load and process the HR image
    hr_image = tf.io.read_file(hr_path)
    hr_image = tf.image.decode_png(hr_image, channels=3)
    hr_image = tf.cast(hr_image, tf.float32) / 255.0

    # Load and process the LR image
    lr_image = tf.io.read_file(lr_path)
    lr_image = tf.image.decode_png(lr_image, channels=3)
    lr_image = tf.cast(lr_image, tf.float32) / 255.0

    lr_patches = []
    hr_patches = []
    # Crop and augment
    for _ in range(num_patches):
      lr_patch, hr_patch = apply_func(lr_image, hr_image, patch_size, scale)

        # Check if augmentation is wanted
      if augment and augment_patch is not None:
        lr_patch, hr_patch = augment_patch(lr_patch, hr_patch)

      lr_patches.append(lr_patch)
      hr_patches.append(hr_patch)

    return tf.stack(lr_patches), tf.stack(hr_patches)

def plot_patches(lr_patches, hr_patches, num_pairs=5):
    plt.figure(figsize=(10, 2 * num_pairs))

    for i in range(num_pairs):
        # Rescale images from 0-1 to 0-255 if they are normalized
        lr_image = np.clip(lr_patches[i].numpy() * 255.0, 0, 255).astype("uint8")
        hr_image = np.clip(hr_patches[i].numpy() * 255.0, 0, 255).astype("uint8")

        # Plot LR patch
        plt.subplot(num_pairs, 2, 2 * i + 1)
        plt.imshow(lr_image)
        plt.title("Low Resolution")
        plt.axis("off")

        # Plot HR patch
        plt.subplot(num_pairs, 2, 2 * i + 2)
        plt.imshow(hr_image)
        plt.title("High Resolution")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
