import cv2
import numpy as np
import os
import time
from PIL import Image

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val != min_val:
        normalized_image = (image - min_val) / (max_val - min_val) * 255.0  # Scale to 0-255
    else:
        normalized_image = image.copy()
    return normalized_image.astype(np.uint8)  # Ensuring the image is uint8 for feature detection

def get_all_channel_files(folder_path):
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    all_channel_files = {}
    for file in tif_files:
        section = file.split('ch')[0]
        if section not in all_channel_files:
            all_channel_files[section] = {}
        channel = int(file.split('ch')[1][:2])
        all_channel_files[section][channel] = file
    return all_channel_files

def robust_feature_matching(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)  # Changed from knnMatch to match as it's suitable for BF with crossCheck
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:30]  # Return top 30 matches for homography

def homography_registration(img1, img2, method=cv2.RANSAC, threshold=5.0):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        raise ValueError("No descriptors found. Check image quality or parameters.")
    matches = robust_feature_matching(des1, des2)
    if len(matches) >= 4:  # Homography needs at least 4 matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, method, threshold)
        return H
    else:
        raise ValueError("Not enough matches found for reliable registration.")

def register_images(img1_ch00, img2_ch00, additional_channels_imgs):
    img1_ch00_normalized = normalize_image(img1_ch00)
    img2_ch00_normalized = normalize_image(img2_ch00)
    H = homography_registration(img2_ch00_normalized, img1_ch00_normalized)
    height, width = img1_ch00.shape[:2]
    registered_images = {'ch00': cv2.warpPerspective(img2_ch00, H, (width, height))}
    for ch, img in additional_channels_imgs.items():
        registered_images[ch] = cv2.warpPerspective(img, H, (width, height))
    return registered_images, H

def main(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print("Error: Input folder does not exist.")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    all_channel_files = get_all_channel_files(input_folder)
    if not all_channel_files:
        print("Error: No images found in the input folder.")
        return

    sorted_sections = sorted(all_channel_files.keys())
    base_section = sorted_sections[0]
    base_channel = 0
    img1_ch00_path = os.path.join(input_folder, all_channel_files[base_section][base_channel])
    img1_ch00 = cv2.imread(img1_ch00_path, cv2.IMREAD_UNCHANGED)

    transformation_file_path = os.path.join(output_folder, 'transformations.txt')
    with open(transformation_file_path, 'w') as f:
        f.write("Transformation Matrices:\n\n")
        for section in sorted_sections[1:]:
            img2_ch00_path = os.path.join(input_folder, all_channel_files[section][base_channel])
            img2_ch00 = cv2.imread(img2_ch00_path, cv2.IMREAD_UNCHANGED)
            additional_channels_imgs = {}
            for ch, file in all_channel_files[section].items():
                if ch != base_channel:
                    additional_channels_imgs[f'ch{ch:02d}'] = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_UNCHANGED)
            registered_imgs, H = register_images(img1_ch00, img2_ch00, additional_channels_imgs)
            f.write(f"Section: {section}\n{H}\n\n")
            for ch, img in registered_imgs.items():
                cv2.imwrite(os.path.join(output_folder, f'{section}-{ch}-registered.tif'), img)

if __name__ == '__main__':
    start_time = time.time()
    main('./input', './output')
    end_time = time.time()
    print(" -------------------------------- Time to run --------------------------------")
    print(end_time - start_time)
