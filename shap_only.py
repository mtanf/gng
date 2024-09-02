
import os
import numpy as np
import argparse as ap
import tensorflow as tf
from tensorflow import keras
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def shap_plot(explainer, X_shap, Y_shap, X_shap_print, num_images_shap, shap_output_path,j):
    slice = X_shap[j:j+num_images_shap]
    slice_print=X_shap_print[j:j+num_images_shap]
    y_slice = Y_shap[j:j+num_images_shap]
    shap_values = explainer(slice,
                            outputs=shap.Explanation.argsort.flip[:num_images_shap],
                            max_evals=10000)

    predictions = model.predict(slice)
    predictions_labels = np.where(predictions > 0.5, 1, 0)

    predictions_to_plot = []
    for i in range(num_images_shap):
        predictions_to_plot.append(
            "Pred:{} | True:{}".format(str(predictions_labels.transpose().tolist()[0][i]),
                                       str(y_slice.transpose().tolist()[0][i])))
    predictions_to_plot = np.array([[i] for i in predictions_to_plot])
    plt.figure()
    shap.image_plot(shap_values, pixel_values=slice_print, labels=predictions_to_plot, show=False)
    if not os.path.isdir(shap_output_path):
        os.makedirs(shap_output_path)
    savepath = os.path.join(shap_output_path, "Shap_results_{}_set{}.png".format(shap_data_origin,j))
    plt.savefig(savepath)
    plt.close("all")
    # matplotlib.pyplot
    # plt.show()

use_data_aug = False
main_model_output_path = os.path.join(os.getcwd(), "gng_old_script")
classifier_name = "MobileNetV3_frozen_useDataAug_" + str(use_data_aug)
weight_output_path = os.path.join(main_model_output_path, classifier_name + "_weights")
model = keras.models.load_model(os.path.join(weight_output_path, classifier_name + "_weights.h5"))

main_shap_output_path = "SHAPOut"
shap_output_path = os.path.join(main_shap_output_path, classifier_name + "_shap_explanations"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if not os.path.isdir(main_shap_output_path):
    os.mkdir(main_shap_output_path)

IMG_WIDTH=224
IMG_HEIGHT=224
BATCH_SIZE=1024
EPOCHS=100

real_train_path = "Real"
generated_train_path = "Fake"
resize_dim = (IMG_WIDTH, IMG_HEIGHT)
model_input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

parent_dir_train = "/homeRepo/tanfoni/Dataset_merged/Train/"
real_dir_train = "Real"
sint_dir_train = "Fake"

parent_dir_val = "/homeRepo/tanfoni/Dataset_merged/Valid/"
real_dir_val = "Real"
sint_dir_val = "Fake"

parent_dir_test = "/homeRepo/tanfoni/Dataset_merged/Test/"
real_dir_test = "Real"
sint_dir_test = "Fake"

dataset_batch_size = BATCH_SIZE
train_batch_size = int(dataset_batch_size * 0.6)
validation_batch_size = int((dataset_batch_size - train_batch_size) * 0.6)
test_batch_size = dataset_batch_size - train_batch_size - validation_batch_size


print("Loading training images")
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_train,
    labels="inferred",
    class_names=[real_dir_train, sint_dir_train],
    label_mode='binary',
    color_mode='rgb',
    batch_size=train_batch_size,
    image_size=resize_dim,
    shuffle=True,
    seed=123,
)

if use_data_aug:
    print("Performing data augmentation on training images")
    data_augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        label_mode='binary',
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_dataset_aug = data_augmentation.flow_from_directory(
        parent_dir_train,
        target_size=resize_dim,
        batch_size=train_batch_size,
        label_mode='binary',
        class_mode='binary',
        shuffle=True,
        seed=123
    )


print("Loading validation images")
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_val,
    labels="inferred",
    class_names=[real_dir_val, sint_dir_val],
    label_mode='binary',
    color_mode='rgb',
    batch_size=validation_batch_size,
    image_size=resize_dim,
    shuffle=True,
    seed=123
)

print("Loading test images")
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_test,
    labels="inferred",
    class_names=[real_dir_test, sint_dir_test],
    label_mode='binary',
    color_mode='rgb',
    batch_size=test_batch_size,
    image_size=resize_dim,
    shuffle=True,
    seed=123
)

num_images_shap = 3
shap_data_origin = "test"

print("Explaining prediction for a validation batch with SHAP for {} images".format(num_images_shap))
print("It might take a while...")

norm_layer = tf.keras.layers.Rescaling(scale=1 / 255)

if shap_data_origin is None or shap_data_origin in ["val", "validation", "v"]:
    shap_print_data = val_dataset.map(lambda x, y: (norm_layer(x), y))
    for images, labels in val_dataset:
        X_shap = images.numpy()
        Y_shap = labels.numpy()
        break

elif shap_data_origin in ["train", "tr"]:
    shap_print_data = train_dataset.map(lambda x, y: (norm_layer(x), y))
    for images, labels in train_dataset:
        X_shap = images.numpy()
        Y_shap = labels.numpy()
        break

elif shap_data_origin in ["test", "t"]:
    shap_print_data = test_dataset.map(lambda x, y: (norm_layer(x), y))
    for images, labels in test_dataset:
        X_shap = images.numpy()
        Y_shap = labels.numpy()
        break

for images, _ in shap_print_data:
    X_shap_print = images.numpy()
    break

masker = shap.maskers.Image("inpaint_telea", X_shap[0].shape)
class_names = ["real", "fake"]

explainer = shap.Explainer(model, masker, output_names=class_names)

if os.path.isdir(shap_output_path):
    for file in os.listdir(shap_output_path):
        os.remove(os.path.join(shap_output_path, file))

for j in range(0,180,num_images_shap):
    shap_plot(explainer, X_shap, Y_shap, X_shap_print, num_images_shap, shap_output_path, j)

