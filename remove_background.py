from utils import *
import keras
from deeplab_mdl_def import DynamicUpsample

"""Deeplabv3+ model for face segmentation.
 The model is trained on the CelebA+FFHQ datasets."""


IMAGE_SIZE = 448

model_name = "deeplabv3plus_face_segmentation_augmentation_class_weights_latest"

model = keras.models.load_model(model_name + ".h5", custom_objects={'DynamicUpsample': DynamicUpsample})

remove_background("/homeRepo/tanfoni/Dataset_merged/", "/homeRepo/tanfoni/Dataset_deeplab/", model, IMAGE_SIZE,
                  close_iterations=4, erode_iterations=0)
