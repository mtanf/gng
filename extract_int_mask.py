import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Defining color maps and translation dictionary
# patch_colors_bgr_01_old = {
#     "background": [0, 0, 0],  # BGR
#     "lips": [0, 0, 1],  # BGR
#     "eyes": [0, 1, 0],  # BGR
#     "nose": [1, 0, 0],  # BGR
#     "face": [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],  # BGR
#     "hair": [0, 1, 1],  # BGR
#     "eyebrow": [1, 0, 1],  # BGR
#     "ears": [1, 1, 0],  # BGR
#     "teeth": [1, 1, 1],  # BGR
#     "facial_hair": [0.7529411764705882, 0.7529411764705882, 1],  # BGR
#     "glasses": [0.5019607843137255, 0.5019607843137255, 0]  # BGR
# }

patch_colors_bgr_01 = {
    "background": [0, 0, 0],  # BGR
    "generalface": [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],  # BGR
    "left_eye": [0, 1, 0],  # BGR
    "right_eye": [0, 0.5019607843137255, 0],  # BGR
    "nose": [1, 0, 0],  # BGR
    "left_ear": [1, 1, 0],  # BGR
    "right_ear": [0.25098039215686274, 0.25098039215686274, 0],  # BGR
    "lips": [0, 0, 1],  # BGR
    "left_eyebrow": [1, 0, 1],  # BGR
    "right_eyebrow": [0.5019607843137255, 0, 0.5019607843137255],  # BGR
    "hair": [0, 1, 1],  # BGR
    "teeth": [1, 1, 1],  # BGR
    "specs": [0.5019607843137255, 0.5019607843137255, 0],  # BGR
    "beard": [0.7529411764705882, 0.7529411764705882, 1]  # BGR
}
patch_colors_rgb_01 = {key: [color[2], color[1], color[0]] for key, color in patch_colors_bgr_01.items()}
# convert patch_colors_rgb_01 to patch_colors_rgb_255
patch_colors_rgb_255 = {}
for key, value in patch_colors_rgb_01.items():
    patch_colors_rgb_255[key] = [int(255 * x) for x in value]

# patch_colors_int_old = {
#     "background": 0,
#     "lips": 1,
#     "eyes": 2,
#     "nose": 3,
#     "face": 4,
#     "hair": 5,
#     "eyebrow": 6,
#     "ears": 7,
#     "teeth": 8,
#     "facial_hair": 9,
#     "glasses": 10
# }

patch_colors_int = {
    "background": 0,
    "generalface": 1,
    "left_eye": 2,
    "right_eye": 3,
    "nose": 4,
    "left_ear": 5,
    "right_ear": 6,
    "lips": 7,
    "left_eyebrow": 8,
    "right_eyebrow": 9,
    "hair": 10,
    "teeth": 11,
    "specs": 12,
    "beard": 13
}

df_translate = pd.DataFrame({
    'Patch': list(patch_colors_int.keys()),
    'RGB_255': list(patch_colors_rgb_255.values()),
    'Integer': list(patch_colors_int.values())
})

print(df_translate)

def get_int_mask(arr, df_translate):
    converted_msk = np.zeros_like(arr, dtype=np.uint8)
    # Loop through the DataFrame and replace colors in the image
    for index, row in df_translate.iterrows():
        patch_color = row['RGB_255']
        integer_value = row['Integer']
        mask = np.all(arr == patch_color, axis=-1)
        converted_msk[mask] = integer_value
    return converted_msk


# convert masks
masks_folder_path = "/homeRepo/tanfoni/faceSegmentation/dataset_paid/masks"
masks_output_folder_path = "/homeRepo/tanfoni/faceSegmentation/dataset_paid_integers/masks"

if not os.path.isdir(masks_output_folder_path):
    os.makedirs(masks_output_folder_path)

for filename in tqdm(os.listdir(masks_folder_path)):
    image_path = os.path.join(masks_folder_path, filename)
    save_cmsk_path = os.path.join(masks_output_folder_path, filename)

    image = Image.open(image_path)
    image = np.asarray(image)

    cmsk = get_int_mask(image, df_translate)
    # turn cmsk into png image
    cmsk = Image.fromarray(cmsk)
    cmsk.save(save_cmsk_path)


print("All done!")
