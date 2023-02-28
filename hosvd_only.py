import numpy as np
import cv2
import os
import tensorly as tl
from keras.utils import normalize
import json
import argparse as ap


def load_images_from_folder(folder, new_img_dim):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        # load
        img = cv2.imread(os.path.join(folder, filename))
        # resize
        img = cv2.resize(img, (new_img_dim, new_img_dim))
        # #show
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if img is not None:
            images.append(img)
    return images, filenames


def tucker_decomposed_imgs(img_list, tucker_rank):
    core_imgs = []
    for img in img_list:
        core, factors = tl.decomposition.tucker(img, rank=tucker_rank)
        core_imgs.append(core)
    return core_imgs


def rescale_cores(core_list):
    rescaled_cores = []
    for item in core_list:
        scaled = normalize(item)
        rescaled_cores.append(scaled)
    return rescaled_cores


def save_matrices(matrices_list, original_filenames, savepath):
    # saves each matrix in a matrix list to a .npy file
    for i in range(len(matrices_list)):
        item = matrices_list[i]
        orig_name = original_filenames[i]
        # remove original file extension from filename
        orig_name = orig_name.split(".")[0]
        new_name = orig_name + "_HOSVD_Core"
        np.save(os.path.join(savepath, new_name), item, allow_pickle=False)
    return


def load_matrices(folder_path):
    reloaded_matrices = []
    filenames = os.listdir(folder_path)
    for filename in filenames:
        reloaded_matrices.append(np.load(os.path.join(folder_path, filename), allow_pickle=False))

    return reloaded_matrices


def str_to_bool(string, argname):
    if isinstance(string, bool):  # check if input is already a boolean
        return string
    else:
        lowercase_string = string.lower()  # convert to lowercase to check for less words
        if lowercase_string in ["true", "yes", "y", "t", "whynot", "1", "ok"]:
            return True
        elif lowercase_string in ["false", "no", "n", "f", "nope", "0" "not"]:
            return False
        else:
            raise ap.ArgumentTypeError("Boolean value expected for {}".format(argname))


parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required=True, help="Path to config json.")

args = vars(parser.parse_args())
json_path = args["path_to_json"]

with open(json_path) as f:
    run_params = json.load(f)

force_hosvd = str_to_bool(run_params["force_hosvd"], "Force HOSVD calculation")

real_imgs = run_params["real"]
generated_imgs = run_params["generated"]

real_list, real_names = load_images_from_folder(real_imgs, 1/run_params["compression"])
generated_list, generated_names = load_images_from_folder(generated_imgs)

