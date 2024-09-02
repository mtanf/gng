# Imports for data preprocessing
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import shutil
import xml.etree.ElementTree as ET


def read_image(image_path, mask=False, resize_img=True, new_image_size=512):
    image = tf_io.read_file(image_path)
    if mask:
        n_channels= 1
    else:
        n_channels = 3
    image = tf_image.decode_png(image, channels=n_channels)
    if resize_img:
        image.set_shape([None, None, n_channels])
        image = tf_image.resize(images=image, size=[new_image_size, new_image_size])

    '''
    if mask:
        image = tf_image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf_image.resize(images=image, size=[image_size, image_size])
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf_image.resize(images=image, size=[image_size, image_size])
    '''
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def load_data_noresize(image_list, mask_list):
    image = read_image(image_list, resize_img=False)
    mask = read_image(mask_list, mask=True, resize_img=False)
    return image, mask

def augment(image, mask, prob=0.5):

    should_augment = tf.random.uniform([]) > prob

    if should_augment:

        p_bright = tf.random.uniform([])
        p_contrast = tf.random.uniform([])
        p_hue = tf.random.uniform([])
        p_saturation = tf.random.uniform([])
        p_flip_lr = tf.random.uniform([])
        p_flip_ud = tf.random.uniform([])
        p_noise = tf.random.uniform([])
        p_random_crop = tf.random.uniform([])
        p_rotate = tf.random.uniform([])

        if p_bright > prob:
            image = tf.image.random_brightness(image, 0.2)
        if p_contrast > prob:
            image = tf.image.random_contrast(image, 0.8, 1.2)
        if p_hue > prob:
            image = tf.image.random_hue(image, 0.2)
        if p_saturation > prob:
            image = tf.image.random_saturation(image, 0.8, 1.2)
        if p_noise > prob:
            image =tf.clip_by_value(image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1), 0.0, 255.0)

        concat_data = tf.concat([image, mask], axis=-1)
        if p_flip_lr > prob:
            concat_data = tf.image.flip_left_right(concat_data)
        if p_flip_ud > prob:
            concat_data = tf.image.flip_up_down(concat_data)
        if p_random_crop > prob:
            concat_data = tf.image.random_crop(concat_data, size=[512, 512, 4])
        if p_rotate > prob:
            concat_data = tf.image.rot90(concat_data, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))

        image_aug = concat_data[:, :, :3]
        mask_aug = concat_data[:, :, 3:]

        return image_aug, mask_aug
    else:
        return image, mask

# def data_generator(image_list, mask_list, batch_size):
#     dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
#
#     dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
#     dataset = dataset.map(augment, num_parallel_calls=tf_data.AUTOTUNE)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     return dataset
def data_generator(image_list, mask_list, batch_size, augment_data=True, resize_image=True):

    dataset_original = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    if resize_image:
        dataset_original = dataset_original.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset_original = dataset_original.map(load_data_noresize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        dataset_augmented = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        if resize_image:
            dataset_augmented = dataset_augmented.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset_augmented = dataset_augmented.map(load_data_noresize, num_parallel_calls=tf.data.AUTOTUNE)
        dataset_augmented = dataset_augmented.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset_original.concatenate(dataset_augmented)
    else:
        dataset = dataset_original

    if resize_image == False:
        dataset = dataset.batch(1, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0), verbose=0)
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes=11):
    colormap = [patch for patch in colormap.values()]
    colormap = np.array(colormap) * 100
    colormap = colormap.astype(np.uint8)

    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    # rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # for l in range(0, n_classes):
    #     idx = mask == l
    #     rgb[idx] = colormap.get(str(l), [0, 0, 0])  # Use .get() to handle missing keys gracefully

    return rgb

def get_overlay(image, mask):
    image = Image.fromarray((image * 255).astype(np.uint8))
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5,3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(display_list[i])
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_predictions(images_list,image_size, colormap,model,n_classes=11):
    for image_file in images_list:
        image = Image.open(image_file)#.resize((image_size, image_size))
        image_tensor = np.array(image)
        prediction_mask = infer(model, image_tensor)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, n_classes=n_classes)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib([image_tensor, prediction_colormap, overlay], figsize=(18,14))

def get_unique_colors(image_path):
    """
    Get a set of unique colors from an image
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure image is in RGB format
        colors = img.getcolors(maxcolors=2 ** 24)  # Get all colors from the image
        unique_colors = {color[1] for color in colors} if colors else set()
    return unique_colors


def unique_colors_in_folder(folder_path):
    """
    Get a combined set of unique colors from all images in a folder
    """
    all_colors = set()
    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            unique_colors = get_unique_colors(image_path)
            all_colors.update(unique_colors)

    return all_colors


def focal_loss_multiclass(alpha=0.25, gamma=2.0, num_classes=2, alpha_tensor=None):
    """
    Focal loss for multiclass segmentation using logits.
    Args:
    - alpha (float or list of floats): Balancing factor for each class.
    - gamma (float): Modulating factor.
    - num_classes (int): Number of classes.
    - alpha_tensor (tf.Tensor): Tensor of shape (num_classes,) specifying alpha values for each class (overrides alpha).

    Returns:
    - loss (function): A loss function taking (y_true, y_logits).
    """

    if alpha_tensor is None:
        if isinstance(alpha, list):
            alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)
        else:
            alpha_tensor = tf.fill((num_classes,), alpha)

    def focal_loss_fixed(y_true, y_logits):
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_logits)

        p_t = tf.nn.softmax(y_logits, axis=-1)

        modulating_factor = tf.pow(1.0 - p_t, gamma)

        alpha_t = tf.gather(alpha_tensor, tf.argmax(y_true, axis=-1))

        focal_loss = tf.reduce_sum(alpha_t * modulating_factor * ce_loss, axis=-1)

        return tf.reduce_mean(focal_loss)

    return focal_loss_fixed

def read_masks_and_compute_weights(directory, normalize=True, background_increase=0.1):
    class_counts = {}
    total_pixels = 0

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                mask = np.array(img)
                for class_value in np.unique(mask):
                    if class_value not in class_counts:
                        class_counts[class_value] = 0
                    class_counts[class_value] += np.sum(mask == class_value)
                total_pixels += mask.size

    cw = {}
    if normalize:
        max_weight = 0

        for class_value, count in class_counts.items():
            cw[class_value] = total_pixels / (len(class_counts) * count)
            if cw[class_value] > max_weight:
                max_weight = cw[class_value]

        for class_value in cw:
            cw[class_value] /= max_weight

        cw[0] += background_increase

        for class_value in cw:
            cw[class_value] += 1

    return cw

def load_and_preprocess_image(image_path,image_size=448):
    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    image_array = np.array(image)
    return image_array

def save_segmented_image(predictions, save_path):

    mask = np.array(predictions != 0)
    segmented_image = np.zeros_like(predictions)
    segmented_image[mask] = 255
    image = Image.fromarray(segmented_image.astype('uint8'))
    image.save(save_path)

def save_image(image_array, save_path):
    image = Image.fromarray(image_array)
    image.save(save_path)


def fill_holes(predictions, close_iterations=4, erode_iterations=2):

    mask = predictions != 0

    mask = mask.astype(np.uint8) * 255

    close_kernel = np.ones((10, 10), np.uint8)
    erode_kernel = np.ones((7, 7), np.uint8)

    closed_mask = mask
    for _ in range(close_iterations):
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, close_kernel)

    eroded_mask = closed_mask
    for _ in range(erode_iterations):
        eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_ERODE, erode_kernel)
    return eroded_mask

def apply_mask_to_image(original_image, mask):

    if len(mask.shape) == 2:
        mask = np.stack([mask]*3, axis=-1)


    masked_image = original_image * (mask.astype(original_image.dtype) // 255)
    return masked_image
def remove_background(dataset_dir, output_dir, model, image_size, close_iterations=5, erode_iterations=5):
    for split in ['Train', 'Test', 'Valid']:
        for condition in ['Real', 'Fake']:
            input_dir = os.path.join(dataset_dir, split, condition)
            output_split_dir = os.path.join(output_dir, split, condition)
            os.makedirs(output_split_dir, exist_ok=True)

            for image_name in tqdm(os.listdir(input_dir), desc=f"Segmenting {split} {condition} images"):
                image_path = os.path.join(input_dir, image_name)
                save_path = os.path.join(output_split_dir, image_name)

                image_tensor = load_and_preprocess_image(image_path, image_size)
                predictions = infer(model, image_tensor)

                filled_predictions = fill_holes(predictions, close_iterations=close_iterations, erode_iterations=erode_iterations)

                original_image = load_and_preprocess_image(image_path, image_size)
                masked_image = apply_mask_to_image(original_image, filled_predictions)

                save_image(masked_image, save_path)



def process_dataset_by_class(dataset_dir, output_dir, model, image_size, num_classes):
    """Process a dataset by class, segmenting each image
    and saving it in the corresponding class folder."""
    for class_id in range(num_classes):
        class_dir = os.path.join(output_dir, f"Dataset_{class_id}")
        for split in ['Train', 'Test', 'Valid']:
            for condition in ['Real', 'Fake']:
                os.makedirs(os.path.join(class_dir, split, condition), exist_ok=True)


    for split in ['Train', 'Test', 'Valid']:
        for condition in ['Real', 'Fake']:
            input_dir = os.path.join(dataset_dir, split, condition)

            for image_name in tqdm(os.listdir(input_dir), desc=f"Processing {split} {condition} images"):
                image_path = os.path.join(input_dir, image_name)
                image_tensor = load_and_preprocess_image(image_path, image_size)
                predictions = infer(model, image_tensor)

                for class_id in range(num_classes):
                    class_predictions = (predictions == class_id).astype(np.uint8)
                    filled_mask = fill_holes(class_predictions)


                    original_image = load_and_preprocess_image(image_path, image_size)


                    masked_image = apply_mask_to_image(original_image, filled_mask)


                    save_path = os.path.join(output_dir, f"Dataset_{class_id}", split, condition, image_name)
                    save_image(masked_image, save_path)


import os
import zipfile
from tqdm import tqdm


def zip_files_with_string_in_name(directory_path, search_string):
    parent_directory = os.path.dirname(directory_path)
    zip_file_path = os.path.join(parent_directory, "filtered_files.zip")

    files_to_add = []
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            if search_string in filename:
                file_path = os.path.join(foldername, filename)
                files_to_add.append(file_path)

    with zipfile.ZipFile(zip_file_path, 'w') as zipf, tqdm(total=len(files_to_add), desc="Adding files") as pbar:
        for file_path in files_to_add:
            zipf.write(file_path, os.path.relpath(file_path, directory_path))
            pbar.update(1)

    print(f"Created ZIP file at: {zip_file_path}")


def reorganize_dataset(dataset_dir, xml_file):
    """Reorganize mut1ny dataset into separate 'images' and 'masks' folders, following the paths in the included XML file."""

    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_index = 0
    mask_index = 0

    items = list(root.findall('.//srcimg')) + list(root.findall('.//labelimg'))  # Cerca elementi direttamente nel root

    for child in tqdm(items, desc="Processing images and masks"):
        original_path = os.path.join(dataset_dir, child.attrib['name'].replace('\\', '/'))
        if child.tag == 'srcimg':
            folder = images_dir
            new_file_name = f'{image_index:04d}.png'
            image_index += 1
        else:
            folder = masks_dir
            new_file_name = f'{mask_index:04d}.png'
            mask_index += 1

        if not os.path.exists(original_path):
            print(f"File not found: {original_path}")
            continue

        new_file_path = os.path.join(folder, new_file_name)

        try:
            with Image.open(original_path) as img:
                img.convert('RGB').save(new_file_path, 'PNG')
        except Exception as e:
            print(f"Failed to convert {original_path} to PNG: {e}")

    print("All files have been reorganized.")

# dataset_dir = '/home/tanfoni/homeRepo/tanfoni/faceSegmentation/mut1ny'
# xml_file_path = os.path.join(dataset_dir, 'training.xml')
#
# reorganize_dataset(dataset_dir, xml_file_path)

#
#
#
# directory_path = '/homeRepo/tanfoni/Dataset_stylegan3_only/Fake'
# search_string = 'stylegan3'
# zip_files_with_string_in_name(directory_path, search_string)

