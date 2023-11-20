import os
from sklearn.model_selection import train_test_split
from shutil import copyfile


def copy_files(source_files, input_folder, dest_folder):
    for file in source_files:
        src_path = os.path.join(input_folder, file)
        dest_path = os.path.join(dest_folder, file.split('_')[0], file)
        copyfile(src_path, dest_path)

def split_dataset(input_folder, output_folder, test_size=0.2, valid_size=0.1, random_seed=42):
    """
    Split a dataset into Train, Test, and Valid sets while maintaining the directory structure.

    Parameters:
    - input_folder (str): Path to the input dataset folder.
    - output_folder (str): Path to the output folder where the split dataset will be saved.
    - test_size (float): Proportion of the dataset to include in the test split.
    - valid_size (float): Proportion of the dataset to include in the valid split.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - None
    """

    # Create output folders
    train_folder = os.path.join(output_folder, 'Train')
    test_folder = os.path.join(output_folder, 'Test')
    valid_folder = os.path.join(output_folder, 'Valid')

    for folder in [train_folder, test_folder, valid_folder]:
        os.makedirs(os.path.join(folder, 'Fake'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'Real'), exist_ok=True)

    # Get list of all files in the input folder
    file_list = os.listdir(input_folder)

    # Separate files into Fake and Real
    fake_files = [file for file in file_list if file.startswith('Fake')]
    real_files = [file for file in file_list if file.startswith('Real')]

    # Split Fake files
    fake_train, fake_test_valid = train_test_split(fake_files, test_size=test_size, stratify=[f.split('_')[1] for f in fake_files], random_state=random_seed)
    fake_test, fake_valid = train_test_split(fake_test_valid, test_size=valid_size/(1-test_size), stratify=[f.split('_')[1] for f in fake_test_valid], random_state=random_seed)

    # Split Real files
    real_train, real_test_valid = train_test_split(real_files, test_size=test_size, stratify=[f.split('_')[1] for f in real_files], random_state=random_seed)
    real_test, real_valid = train_test_split(real_test_valid, test_size=valid_size/(1-test_size), stratify=[f.split('_')[1] for f in real_test_valid], random_state=random_seed)

    # Copy files to the output folders
    copy_files(fake_train, input_folder, train_folder)
    copy_files(fake_test, input_folder, test_folder)
    copy_files(fake_valid, input_folder, valid_folder)

    copy_files(real_train, input_folder, train_folder)
    copy_files(real_test, input_folder, test_folder)
    copy_files(real_valid, input_folder, valid_folder)


# Example usage:
input_folder_path = '/path/to/your/input/dataset'
output_folder_path = '/path/to/your/output/folder'
split_dataset(input_folder_path, output_folder_path)
