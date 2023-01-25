import os
import random

import os
import random
import shutil

random.seed(48) # nice seeds for template: 45, 48

# specify the source folder
source_folder = os.path.join("dataset", "Real_run_4K")
tag = "_real"
# specify the destination folders

template_dir = os.path.join("Dataset","template" + tag)
train_dir = os.path.join("Dataset","train" + tag)
valid_dir = os.path.join("Dataset","validation" + tag)
test_dir = os.path.join("Dataset","test" + tag)

destination_folders = [template_dir,train_dir,valid_dir,test_dir]

# create non-existing folders and clear previously created ones
for paths in destination_folders:
    if not (os.path.isdir(paths)):
        os.mkdir(paths)
    else:
        shutil.rmtree(paths)
        os.mkdir(paths)

# count the number of elements in the folders

jpg_count = len([file for file in os.listdir(source_folder) if file.endswith(".jpg")])
png_count = len([file for file in os.listdir(source_folder) if file.endswith(".png")])

print(f"Numero di file .jpg: {jpg_count}, Numero di file .png: {png_count}")
img_count = jpg_count + png_count

# specify the number of files to select
num_template = 40
num_files_to_select = img_count - 40
num_train = int(num_files_to_select * 0.7)
num_valid = int((num_files_to_select - num_train) * 0.33)
num_test = num_files_to_select - num_train - num_valid

# get the list of files in the source folder
files = os.listdir(source_folder)




def mover (selected_files, source_folder, destination_folder,files_list):
    for random_file in selected_files:
        # build the full path of the randomly selected file
        random_file_path = os.path.join(source_folder, random_file)

        # build the full path of the destination folder
        destination_path = os.path.join(destination_folder, random_file)

        # move the randomly selected file to the destination folder
        shutil.copy(random_file_path, destination_path)

    files_list = [x for x in files_list if x not in selected_files]
    return files_list

print("n. of template samples:", num_template)
print("n. of train samples:", num_train)
print("n. of validation samples:", num_valid)
print("n. of test samples:", num_test)

# randomly select and copy TEMPLATE imgs to the destination folder
selected_templates = random.sample(files, num_template)
files = mover(selected_templates,source_folder,template_dir,files)

# randomly select and copy TRAIN imgs from the remaining ones to the destination folder
selected_train = random.sample(files, num_train)
files = mover(selected_train,source_folder,train_dir,files)

# randomly select and copy VALIDATION imgs from the remaining ones to the destination folder
selected_validation = random.sample(files, num_valid)
files = mover(selected_validation,source_folder,valid_dir,files)

# randomly select and copy TEST imgs from the remaining ones to the destination folder
selected_test = random.sample(files, num_test)
files = mover(selected_test,source_folder,test_dir,files)







