import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
from tqdm import tqdm
import random
from deeplab_mdl_def import DynamicUpsample
import matplotlib.pyplot as plt

# Suppress interactive plot display
plt.ioff()

# Paths for datasets
with_bg_base_path = "/homeRepo/tanfoni/Dataset_sg2"
without_bg_base_path = "/homeRepo/tanfoni/Dataset_sg2_no_background"
model_path = "/homeRepo/tanfoni/keras_deeplab_faceseg/Results/Deeplab+fake_detection/trained_models/deeplab+fake_detection_transfer_learning_Transfer_learning_last_xception.h5"

# Model loading
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = keras.models.load_model(model_path, custom_objects={'DynamicUpsample': DynamicUpsample})

# SHAP setup
num_images_shap = 50  # Number of images to analyze per split
random_seed = 42  # Seed for reproducibility
shap_output_path = "/homeRepo/tanfoni/keras_deeplab_faceseg/shap_comparison_bg_nobg"
if not os.path.isdir(shap_output_path):
    os.makedirs(shap_output_path)

# Prepare data
resize_dim = (224, 224)

def load_images_from_dir(dir_path, file_list):
    """Load and preprocess images from a directory based on a provided file list."""
    images = []
    for file in file_list:
        img_path = os.path.join(dir_path, file)
        if os.path.exists(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=resize_dim)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)
    return np.array(images)

# Match filenames between datasets
for split in ["Train", "Valid", "Test"]:
    with_bg_dir = os.path.join(with_bg_base_path, split, "Fake")
    without_bg_dir = os.path.join(without_bg_base_path, split, "Fake")

    with_bg_files = set(os.listdir(with_bg_dir))
    without_bg_files = set(os.listdir(without_bg_dir))

    matched_files = list(with_bg_files & without_bg_files)

    print(f"{len(matched_files)} matching files found in {split}/Fake set.")

    # Randomly select a subset of files based on the seed
    random.seed(random_seed)
    selected_files = random.sample(matched_files, min(num_images_shap, len(matched_files)))

    # Load images with and without background
    with_bg_images = load_images_from_dir(with_bg_dir, selected_files)
    without_bg_images = load_images_from_dir(without_bg_dir, selected_files)

    # Perform SHAP analysis for the selected images
    masker = shap.maskers.Image("inpaint_telea", with_bg_images[0].shape)
    explainer = shap.Explainer(model, masker, output_names=["Real", "Fake"])

    for j in tqdm(range(len(selected_files)), desc=f"Processing {split}/Fake set"):
        img_with_bg = with_bg_images[j:j + 1]
        img_without_bg = without_bg_images[j:j + 1]

        # Make predictions
        pred_with_bg = model.predict(img_with_bg)
        pred_without_bg = model.predict(img_without_bg)

        # Convert predictions to labels
        pred_label_with_bg = "Fake" if pred_with_bg[0][0] > 0.5 else "Real"
        pred_label_without_bg = "Fake" if pred_without_bg[0][0] > 0.5 else "Real"

        # True label (in questo caso "Fake")
        true_label = "Fake"

        # Calculate SHAP values
        shap_values_with_bg = explainer(img_with_bg, max_evals=5000, outputs=shap.Explanation.argsort.flip[:1])
        shap_values_without_bg = explainer(img_without_bg, max_evals=5000, outputs=shap.Explanation.argsort.flip[:1])

        # Save SHAP results
        save_path_with_bg = os.path.join(shap_output_path, f"shap_with_bg_{split.lower()}_{j + 1}.npy")
        save_path_without_bg = os.path.join(shap_output_path, f"shap_without_bg_{split.lower()}_{j + 1}.npy")

        np.save(save_path_with_bg, shap_values_with_bg.values)
        np.save(save_path_without_bg, shap_values_without_bg.values)

        # Save visualizations with "Pred: ... | True: ..."
        vis_path_with_bg = os.path.join(shap_output_path, f"shap_with_bg_{split.lower()}_{j + 1}.png")
        vis_path_without_bg = os.path.join(shap_output_path, f"shap_without_bg_{split.lower()}_{j + 1}.png")

        plt.figure(figsize=(10, 5))
        shap.image_plot(shap_values_with_bg, img_with_bg, show=False)
        plt.title(f"Pred: {pred_label_with_bg} | True: {true_label}")
        plt.savefig(vis_path_with_bg)
        plt.close()

        plt.figure(figsize=(10, 5))
        shap.image_plot(shap_values_without_bg, img_without_bg, show=False)
        plt.title(f"Pred: {pred_label_without_bg} | True: {true_label}")
        plt.savefig(vis_path_without_bg)
        plt.close()

print("SHAP analysis completed.")
