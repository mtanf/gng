# merge_dataset.py
import os
import shutil



def merge_dataset(source_folder, output_folder):
    """
    Merges images from subfolders within "Fake" and "Real"
    based on the source model or dataset.
    It recursively traverses all subfolders, regardless of their depth,
    and copies images into "Dataset_merged/Fake" and "Dataset_merged/Real" with unique names.
    The new image names include a prefix to indicate
    the source and a unique incrementing identifier (e.g., "source_name.jpg")."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for source_label in ["Fake", "Real"]:
        source_label_folder = os.path.join(source_folder, source_label)
        output_label_folder = os.path.join(output_folder, source_label)

        if not os.path.exists(output_label_folder):
            os.makedirs(output_label_folder)

        for model_or_dataset in os.listdir(source_label_folder):
            model_or_dataset_source_folder = os.path.join(source_label_folder, model_or_dataset)

            for root, _, files in os.walk(model_or_dataset_source_folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        source_path = os.path.join(root, file)
                        filename, file_extension = os.path.splitext(file)
                        new_filename = f"{model_or_dataset}_{os.path.basename(source_path)}{file_extension}"
                        output_path = os.path.join(output_label_folder, new_filename)
                        shutil.copyfile(source_path, output_path)


