import numpy as np
import cv2
import os
import tensorly as tl
from keras.utils import normalize
import json
import argparse as ap
from tqdm import tqdm

def load_images_from_folder(folder, new_img_dim):
    images = []
    filenames = os.listdir(folder)
    for filename in tqdm(filenames, desc="Loaded"):
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
    print("Tucker decomposition...")
    for i in tqdm(range(len(img_list)), desc='Decomposed'):
        img = img_list[i]
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

hosvd_savefolder = run_params["hosvd_savefolder"]

if not os.path.isdir(hosvd_savefolder):
    force_hosvd = True #savefolder did not exist, hosvd was not calculated/did not complete
    os.mkdir(hosvd_savefolder)


real_imgs = run_params["real"]
generated_imgs = run_params["generated"]
real_hosvd_path = os.path.join(hosvd_savefolder, os.path.basename(real_imgs))
generated_hosvd_path = os.path.join(hosvd_savefolder, os.path.basename(generated_imgs))

if not os.path.isdir(real_hosvd_path):
    os.mkdir(real_hosvd_path)

if not os.path.isdir(generated_hosvd_path):
    os.mkdir(generated_hosvd_path)

#usage parameters
img_dim = run_params["resize_img_dim"]
compression = run_params["compression"]
hosvd_channels = run_params["hosvd_channels"]

d1 = int(img_dim/compression)
tucker_rank = [d1,d1,hosvd_channels]
# tucker_rank = run_params["tucker_rank"]
print("Loading real images...")
real_list, real_names = load_images_from_folder(real_imgs, img_dim)
print("Loading generated images...")
generated_list, generated_names = load_images_from_folder(generated_imgs, img_dim)

print("Peforming HOSVD with Tucker decomposition\nCore tensors will be rescaled with L2 norm")
print("Real images")
#decompose
real_cores = tucker_decomposed_imgs(real_list, tucker_rank)
#rescale
real_cores = rescale_cores(real_cores)
#saving template cores
save_matrices(real_cores, real_names, real_hosvd_path)
print("Generated images")
#decompose
gen_cores = tucker_decomposed_imgs(generated_list, tucker_rank)
#rescale
gen_cores = rescale_cores(gen_cores)
#saving template cores
save_matrices(gen_cores, generated_names, generated_hosvd_path)