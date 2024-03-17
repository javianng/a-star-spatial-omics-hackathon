# Code Sources:
# (1) https://stackoverflow.com/a/10404957
# (2) ChatGPT 3.5

import time
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from matplotlib import pyplot

# VERY DANGEROUS NEXT STEP
# We are doing this because the high-res image is supremely huge
#########################################
Image.MAX_IMAGE_PIXELS = None # DANGEROUS
#########################################

def crop_and_save_tif(input_path, output_path):
    """
    Takes only the center 2400px by 2400px of the image
    """
    # Load the image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the coordinates for cropping
    start_x, start_y = (width - 2400) // 2, (height - 2400) // 2
    end_x, end_y = start_x + 2400, start_y + 2400

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Save the cropped image as a .tif file
    cv2.imwrite(output_path, cropped_image)

def make_grayscale(filepath):
    img = Image.open(filepath)
    img.getdata()
    r, g, b = img.split()
    ra, ga, ba = np.array(r), np.array(g), np.array(b)
    grayscale_as_list = ((0.299 * ra) + (0.587 * ga) + (0.114 * ba))
    return grayscale_as_list

def print_grayscale(grayscale_values):
    pyplot.figure()
    pyplot.imshow(grayscale_values, cmap="gray")
    
def grayscale_to_image(grayscale_values):
    height, width = len(grayscale_values), len(grayscale_values[0])
    img = Image.new("L", (width, height))

    # Iterate through the nested list and set pixel values
    for y in range(height):
        for x in range(width):
            pixel_value = round(grayscale_values[y][x])
            img.putpixel((x, y), pixel_value)

    return img

def mse(image1, image2):
    # Convert images to numpy arrays
    array1 = np.array(image1)
    array2 = np.array(image2)
    
    # Compute the Mean Squared Error (MSE)
    mse_value = np.mean((array1 - array2) ** 2)
    
    return mse_value

def mse_big_small(fidpath, hrpath):
    fid = Image.open(fidpath)
    fid.getdata()
    fid_red_data = np.array(fid.split()[0])
    fid_height, fid_width = len(fid_red_data), len(fid_red_data[0])
    
    hr = Image.open(hrpath)
    hr = hr.resize((fid_width, fid_height))
    return mse(make_grayscale(fidpath), hr)

def g2g(fidpath, hrpath):
    fid = Image.open(fidpath)
    hr = Image.open(hrpath)
    hr = hr.resize(fid.size)
    return mse(fid, hr)

def image_production(filepath):
    grayscale_image = grayscale_to_image(make_grayscale(filepath))
    mirrored_grayscale = grayscale_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    # Get the directory path of the Jupyter notebook
    notebook_dir = os.path.dirname(os.path.abspath("__file__"))
    
    # Create a folder with the same name as the filepath
    folder_name = os.path.join(notebook_dir, os.path.splitext(os.path.basename(filepath))[0])
    os.makedirs(folder_name, exist_ok=True)
    
    new_path_names = []
    
    for degree in [0, 90, 180, 270]:
        rotated_image = grayscale_image.rotate(degree)
        new_os_pathname = os.path.basename(filepath)[:-4] + f"_original_r{degree}.tif"
        rotated_image_path = os.path.join(folder_name, new_os_pathname)
        new_path_names.append(rotated_image_path)
        
        rotated_mirrored_image = mirrored_grayscale.rotate(degree)
        new_os_pathname = os.path.basename(filepath)[:-4] + f"_mirrored_r{degree}.tif"
        rotated_mirrored_image_path = os.path.join(folder_name, new_os_pathname)
        new_path_names.append(rotated_mirrored_image_path)
        
        rotated_image.save(rotated_image_path)
        rotated_mirrored_image.save(rotated_mirrored_image_path)
        
    print(f"Rotated images saved at: {filepath[:-4]}")
    return new_path_names

def find_most_similar_image(large_image_path, small_images_paths):
    # Load the first small image to get its size
    sample_small_image = cv2.imread(small_images_paths[0], cv2.IMREAD_GRAYSCALE)

    # Load the large image
    large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the large image to match the size of the small image
    large_image_resized = cv2.resize(large_image, (sample_small_image.shape[1], sample_small_image.shape[0]))

    # Initialize variables to store the best match and its similarity score
    best_match = None
    best_score = -1

    # Iterate through each small image
    for small_image_path in small_images_paths:
        print("starting...")
        # Load the small image
        small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

        # Calculate the structural similarity index
        score, _ = ssim(large_image_resized, small_image, full=True)

        # If the similarity score is higher than the current best score, update the best match
        if score > best_score:
            best_score = score
            best_match = small_image_path
        print("finished 1!")

    return best_match, best_score

def find_orientation(fid_filepath: str, hr_filepath: str):
    crop_and_save_tif(fid_filepath, "temp_cropped_fid.tif")
    little_files = image_production("temp_cropped_fid.tif")
    return find_most_similar_image(hr_filepath, little_files)


#################################
# USAGE USAGE USAGE USAGE USAGE #
#################################

fiducial_frame_path = "CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_image.tif"
high_res_path = "CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_tissue_image.tif"

start_time = time.time()

fin = find_orientation(fiducial_frame_path, high_res_path)
print(fin)

end_time = time.time()

print(end_time - start_time)