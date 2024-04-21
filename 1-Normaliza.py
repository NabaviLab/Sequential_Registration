from PIL import Image
import os
import numpy as np

def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    if max_val != min_val:
        image = (image - min_val) / (max_val - min_val) * 65535
    return image.astype('uint16')

def process_images(source_folder, target_folder):
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.tif'):
            file_path = os.path.join(source_folder, file_name)
            image = Image.open(file_path)
            image_array = np.array(image)
            normalized_image_array = normalize_image(image_array)
            normalized_image = Image.fromarray(normalized_image_array)
            save_path = os.path.join(target_folder, file_name)
            normalized_image.save(save_path, format='TIFF')

source_folder = './'
target_folder = './'
process_images(source_folder, target_folder)
