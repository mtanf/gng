import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm


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
