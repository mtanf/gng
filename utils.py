# merge_dataset.py



# merge_dataset.py
import os
import shutil
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
from multiprocessing import Pool, Manager
from shutil import copyfile
from sklearn.model_selection import train_test_split

def merge_dataset(source_folder, output_folder):
    """
                Merges images from subfolders within "Fake" and "Real"
                based on the source model or dataset.
                It recursively traverses all subfolders, regardless of their depth,
                and copies images into "Dataset_merged/Fake" and "Dataset_merged/Real" with unique names.
                The new image names include a prefix to indicate
                the source and a unique incrementing identifier (e.g., "source_name.jpg")."""
    types = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for source_label in ["Fake", "Real"]:
        source_label_folder = os.path.join(source_folder, source_label)
        output_label_folder = os.path.join(output_folder, source_label)

        if not os.path.exists(output_label_folder):
            os.makedirs(output_label_folder)

        for model_or_dataset in os.listdir(source_label_folder):
            model_or_dataset_source_folder = os.path.join(source_label_folder, model_or_dataset)
            i=0
            for root, _, files in os.walk(model_or_dataset_source_folder):
                i+=1
                with tqdm(total=len(files), desc=f'Merging {model_or_dataset}', leave=False) as pbar:
                    for file in files:
                        types.append(file.split('.')[-1])
                        if not file.lower().endswith(('.csv', '.txt')):
                            source_path = os.path.join(root, file)
                            file_extension = os.path.splitext(file)[-1]
                            new_filename = f"{model_or_dataset.replace(' ', '_')}_{'folder'}_{i}_{pbar.n:04d}{file_extension}"
                            output_path = os.path.join(output_label_folder, new_filename)
                            shutil.copyfile(source_path, output_path)
                            pbar.update(1)


def apply_grabcut_to_images(input_folder, output_folder):
    """
    Apply the GrabCut algorithm to the images in the input folder and save the results in the output folder.

    :param input_folder: Path to the folder containing the images to process.
    :param output_folder: Path to the folder to save the processed images.
    """
    # Create an output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(input_folder, filename)

            # Load the image
            image = cv2.imread(img_path)

            # Check if the image was loaded successfully
            if image is not None:
                # Clone the input image to work on
                image_copy = image.copy()

                # Create a rectangle that covers the entire image
                h, w, _ = image_copy.shape
                rect = (1, 1, w - 1, h - 1)

                # Initialize the mask and background/foreground model
                mask = np.zeros(image.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)

                # Execute GrabCut
                cv2.grabCut(image_copy, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

                # Modify the mask to obtain the desired result
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

                # Multiply the original image by the obtained mask
                result = image * mask2[:, :, np.newaxis]

                # Save the resulting image in the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, result)

                print(f"Processed: {filename}")
    print("Processing completed.")

def process_image(input_path, output_path):
    detector = MTCNN()
    image = cv2.imread(input_path)
    faces = detector.detect_faces(image)

    for i, face_info in enumerate(faces):
        x, y, width, height = face_info['box']
        face = image[y:y + height, x:x + width]
        # Resize the face to 200x200
        face = cv2.resize(face, (200, 200))
        face_output_path = output_path.replace(".", f"_face_{i}.")
        cv2.imwrite(face_output_path, face)
        print(f"Face {i + 1} extracted from {input_path} and saved to {face_output_path}")


def MTCNN_extractor_multiprocessing(input_dir, output_dir, num_processes = 16):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    output_paths = [os.path.join(output_dir, filename) for filename in os.listdir(input_dir)]

    with Pool(num_processes) as pool:
        pool.imap_unordered(process_image, zip(input_paths, output_paths), chunksize=1000)
        pool.close()
        pool.join()
        pool.terminate()

        # #this is to avoid memory leaks in the pool processes
        # del pool

def MTCNN_extractor(input_dir, output_dir):
    # Initialize the MTCNN face detector
    detector = MTCNN()

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process images in the input directory
    for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load the image
        image = cv2.imread(input_path)

        # Use MTCNN to detect faces in the image
        faces = detector.detect_faces(image)

        for i, face_info in enumerate(faces):
            # Extract the face rectangle
            x, y, width, height = face_info['box']

            # Crop the face from the original image
            face = image[y:y+height, x:x+width]

            # Generate a unique output path for each face
            face_output_path = output_path.replace(".", f"_face_{i}.")

            # Save the face in the output
            cv2.imwrite(face_output_path, face)

            print(f"Face {i+1} extracted from {input_path} and saved to {face_output_path}")

def split_and_copy(source_folder, output_folder, category):
    """
    Split files in a category into Train, Test, and Valid sets and copy them to the output folder.

    Parameters:
        - source_folder (str): Path to the source dataset folder.
        - output_folder (str): Path to the output folder where the split dataset will be saved.
        - category (str): Category of the files (e.g., "Fake" or "Real").

    Returns:
        None
    """
    # Define input and output paths
    input_path = os.path.join(source_folder, category)
    output_category_path = os.path.join(output_folder, category)
    os.makedirs(output_category_path, exist_ok=True)

    # Split files into Train, Test, and Valid sets
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    criteria = [f.split('_')[0] for f in files]
    train_files, test_valid_files, criteria_train, criteria_test_valid = train_test_split(files,
                                                                                          criteria,
                                                                                          test_size=0.3,
                                                                                          random_state=42,
                                                                                          stratify=criteria)
    test_files, valid_files, criteria_test, criteria_valid = train_test_split(test_valid_files,
                                                                              criteria_test_valid,
                                                                              test_size=0.33,
                                                                              # 20% del dataset originale
                                                                              random_state=42,
                                                                              stratify=criteria_test_valid)

    # Create Train, Test, and Valid folders
    for folder in ["Train", "Test", "Valid"]:
        os.makedirs(os.path.join(output_category_path, folder), exist_ok=True)
    # Print class distribution for each set, adding a new line for each category for readability. Also count the number of
    # files in the final split dataset.
    print(f"Train: {len(train_files)} files")
    print(f"Test: {len(test_files)} files")
    print(f"Valid: {len(valid_files)} files\n")
    print(f"Total: {len(train_files) + len(test_files) + len(valid_files)} files\n")

    # count and print criteria_train, criteria_test, criteria_valid occurences for each element
    print (f"Train classes and count:{dict((x,criteria_train.count(x)) for x in set(criteria_train))}")
    print (f"Test classes and count:{dict((x,criteria_test.count(x)) for x in set(criteria_test))}")
    print (f"Valid classes and count:{dict((x,criteria_valid.count(x)) for x in set(criteria_valid))}")

    # Copy files to output folders
    for file in tqdm(train_files, desc=f"Copying {category} Train files"):
        copyfile(os.path.join(input_path, file), os.path.join(output_category_path, "Train", file))

    for file in tqdm(test_files, desc=f"Copying {category} Test files"):
        copyfile(os.path.join(input_path, file), os.path.join(output_category_path, "Test", file))

    for file in tqdm(valid_files, desc=f"Copying {category} Valid files"):
        copyfile(os.path.join(input_path, file), os.path.join(output_category_path, "Valid", file))


def split_dataset(input_folder, output_folder):
    """
    Split the entire dataset into Train, Test, and Valid sets for each category.

    Parameters:
        - input_folder (str): Path to the input dataset folder.
        - output_folder (str): Path to the output folder where the split dataset will be saved.

    Returns:
        None
    """
    # Create the main output folder
    os.makedirs(output_folder, exist_ok=True)

    # Specify categories to process
    categories = ["Fake", "Real"]

    # Split and copy files for each category
    for category in categories:
        split_and_copy(input_folder, output_folder, category)


# Example usage
input_folder = "Dataset_mtcnn+deeplab"
output_folder = "Dataset_mtcnn+deeplab_split"
split_dataset(input_folder, output_folder)
input_dir_real = "Dataset_merged/Real"
input_dir_fake = "Dataset_merged/Fake"
output_dir_real = "Dataset_mtcnn/Real_prove"
output_dir_fake = "Dataset_mtcnn/Fake"

# merge_dataset("Dataset/Artifact dataset", "Dataset_merged")
# apply_grabcut_to_images("Dataset_merged/Real", "Dataset_grabcut/Real")
# MTCNN_extractor(input_dir_fake, output_dir_fake)
MTCNN_extractor_multiprocessing(input_dir_real, output_dir_real)

