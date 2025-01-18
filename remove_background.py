from utils import *
import keras
from deeplab_mdl_def import DynamicUpsample

"""Deeplabv3+ model for face segmentation.
 The model is trained on the CelebA+FFHQ datasets."""


IMAGE_SIZE = 448

model_name = "/homeRepo/tanfoni/keras_deeplab_faceseg/Results/Deeplab/models/deeplabv3plus_face_segmentation_pro_Aug_True_mobilenetv3_2024-04-28_23-31-01/deeplabv3plus_face_segmentation_pro_Aug_True_mobilenetv3"

model = keras.models.load_model(model_name + ".h5", custom_objects={'DynamicUpsample': DynamicUpsample})

remove_background("/homeRepo/tanfoni/Dataset_sg3/", "/homeRepo/tanfoni/Dataset_sg3_no_background/", model, IMAGE_SIZE,
                  close_iterations=5, erode_iterations=0)
