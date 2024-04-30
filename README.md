# Sequential_Registration
 
This repository contains Python scripts for performing image registration using OpenCV. The toolkit includes methods for both homography-based and affine transformation-based registration, supporting both rigid and non-rigid transformations. These scripts are designed to handle images in TIFF format, commonly used in scientific imaging.

## Features
1. Normalization: Custom normalization for 16-bit images to enhance feature detection.
2. Robust Feature Matching: Uses AKAZE keypoints for reliable matching.
3. Homography Registration: Applies transformations based on homography matrix computation.
4. Rigid Registration: Utilizes affine transformations for registration.
5. Preservation of Metadata: Original images are used for the final transformation to preserve metadata.

## Prerequisites
Before you start, ensure you have the following installed:
1. Python 3.6 or higher
2. OpenCV
3. NumPy
4. PIL

You can install the necessary Python packages using pip:

```bash
pip install numpy opencv-python pillow
```

## Structure
The repository contains two main scripts:
1. global_Merged.py: Script for registering images based on homography.
2. rigid_registration.py: Script for registering images using affine transformations.

## Usage
To use these scripts, follow these steps:
1. Clone the repository:
```bash
[git clone https://github.com/your-username/image-registration-toolkit.git](https://github.com/NabaviLab/Sequential_Registration.git)
```
2. Place your TIFF images in an input directory.
3. Run the desired registration script. For example, to perform homography-based registration, use:
```bash
python global_Merged.py --input ./input --output ./output
```
To perform rigid registration, use:
```bash
python affine_Merged.py --input ./input --output ./output
```
## Parameters
1. --input: Directory containing the input TIFF images.
2. --output: Directory where the registered images will be saved.

