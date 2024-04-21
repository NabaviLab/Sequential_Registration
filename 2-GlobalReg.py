import cv2
import numpy as np
import os
import time

def normalize_image(image):
    bit_depth = image.dtype
    if bit_depth == np.uint8:
        normalized_image = image
    elif bit_depth == np.uint16:
        normalized_image = image
    else:
        raise ValueError("Unsupported image bit-depth for normalization.")
    return normalized_image

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
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def homography_registration(img1, img2, method=cv2.LMEDS, threshold=200.0):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    matches = robust_feature_matching(des1, des2)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, method, threshold)
    return H

def register_images(img1_ch00, img2_ch00, additional_channels_imgs):
    img1_ch00_normalized = normalize_image(img1_ch00)
    img2_ch00_normalized = normalize_image(img2_ch00)  
    height, width = img2_ch00_normalized.shape[:2]
    H = homography_registration(img2_ch00_normalized, img1_ch00_normalized)
    registered_images = {'ch00': cv2.warpPerspective(img2_ch00_normalized, H, (width, height))}
    for ch, img in additional_channels_imgs.items():
        if img is not None:
            normalized_img = normalize_image(img)
            registered_images[ch] = cv2.warpPerspective(normalized_img, H, (width, height))
    return registered_images, H

def get_image_files(folder_path, channel):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_files = [f for f in files if f.endswith('.tif') and f.endswith(f'ch{channel:02d}.tif')]
    return sorted(image_files)


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
    
    input_folder_path = './'
    output_folder_path = './'
    
    main(input_folder_path, output_folder_path)
    
    end_time = time.time()
    algorithm_time = end_time - start_time
    print(" -------------------------------- Time to run --------------------------------")
    print(algorithm_time)