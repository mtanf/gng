import os
import time

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from deeplab_mdl_def import DynamicUpsample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.metrics import roc_curve, auc

# Set the GPU depending on the machine
machine_name = "anubis"  # "anubis" or "mec-ai" or "apophis"
if machine_name == "apophis" or "anubis":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    home_path = "/repo/tanfoni/"
    repo_path = home_path + "kerasDeeplabFaceseg/"
elif machine_name == "mec-ai":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    home_path = "/homeRepo/tanfoni/"
    repo_path = home_path + "keras_deeplab_faceseg/"

BACKBONE = "resnet50"  # "resnet50" or "mobilenetv3"


def DeeplabV3Plus_nicco(num_classes,
                        filters_conv1=24, filters_conv2=24,
                        filters_spp=128, filters_final=128,
                        dilated_conv_rates=[1, 4, 8, 16],
                        trainable_resnet=True,
                        path_model_trained: str = None,
                        transfer_learning: str = "last"):
    assert transfer_learning in ["last", "all"], "NICCO: Transfer learning policy not supported"
    model_input = keras.Input(shape=(None, None, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=preprocessed)
    resnet50.trainable = trainable_resnet

    x = resnet50.get_layer("conv4_block6_2_relu").output
    input_b = resnet50.get_layer("conv2_block3_2_relu").output

    x1 = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Reshape((1, 1, x.shape[-1]))(x1)
    x1 = layers.Conv2D(filters=filters_conv1, kernel_size=1, padding="same")(x1)  # Perché qui c'è una conv2d?
    x1 = layers.BatchNormalization()(x1)
    x1 = DynamicUpsample()(x1, x)

    # Multiple dilated convolutions
    pyramids = []

    for rate in dilated_conv_rates:
        if rate == 1:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3, dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)
        else:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3 + int(rate * (1 / 3)), dilation_rate=rate,
                                    padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)

    x = layers.Concatenate(axis=-1)([x1] + pyramids)
    x = layers.Conv2D(filters=filters_spp, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    input_b = layers.Conv2D(filters=filters_conv2, kernel_size=1, padding="same")(input_b)
    input_b = layers.BatchNormalization()(input_b)

    input_a = DynamicUpsample()(x, input_b)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = DynamicUpsample()(x, model_input)

    # resnet_output = resnet50.get_layer("conv4_block6_2_relu").output
    # resnet_output = layers.GlobalAveragePooling2D()(resnet_output)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Concatenate(axis=-1)([x, resnet_output])
    x = layers.Dense(200, activation="selu")(x)
    model_output = layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs=model_input, outputs=model_output)

    if path_model_trained is not None:
        model_from = tf.keras.models.load_model(path_model_trained,
                                                custom_objects={'DynamicUpsample': DynamicUpsample})
        for l, l_old in zip(model.layers[:-3], model_from.layers):
            # Print layer shapes
            # print("Layer: ", l.name, l.trainable, "new", [i.shape for i in l.get_weights()], "old", [i.shape for i in l_old.get_weights()])
            try:
                l.set_weights(l_old.get_weights())
            except Exception as e:
                print("Error in Layer: ", l.name, l.trainable, "new", [i.shape for i in l.get_weights()], "old",
                      [i.shape for i in l_old.get_weights()], e)
            if transfer_learning == "last": l.trainable = False
    resnet50.trainable = trainable_resnet

    return model


def DeeplabV3Plus_nicco_mobilenetv3(num_classes,
                                    filters_conv1=24, filters_conv2=24,
                                    filters_spp=128, filters_final=128,
                                    dilated_conv_rates=[1, 4, 8, 16],
                                    trainable_backbone=True,
                                    path_model_trained: str = None,
                                    transfer_learning: str = "last"):
    assert transfer_learning in ["last", "all"], "Transfer learning policy not supported"
    model_input = keras.Input(shape=(None, None, 3))
    preprocessed = keras.applications.mobilenet_v3.preprocess_input(model_input)
    mobilenetv3 = keras.applications.MobileNetV3Large(weights="imagenet", include_top=True, input_tensor=preprocessed)
    mobilenetv3.trainable = trainable_backbone

    x = mobilenetv3.get_layer("expanded_conv_12/project").output
    input_b = mobilenetv3.get_layer("expanded_conv_3/project").output

    x1 = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Reshape((1, 1, x.shape[-1]))(x1)
    x1 = layers.Conv2D(filters=filters_conv1, kernel_size=1, padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = DynamicUpsample()(x1, x)

    pyramids = []
    for rate in dilated_conv_rates:
        if rate == 1:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3, dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)
        else:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3 + int(rate * (1 / 3)), dilation_rate=rate,
                                    padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)

    x = layers.Concatenate(axis=-1)([x1] + pyramids)
    x = layers.Conv2D(filters=filters_spp, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    input_b = layers.Conv2D(filters=filters_conv2, kernel_size=1, padding="same")(input_b)
    input_b = layers.BatchNormalization()(input_b)

    input_a = DynamicUpsample()(x, input_b)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = DynamicUpsample()(x, model_input)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(200, activation="selu")(x)
    model_output = layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs=model_input, outputs=model_output)

    if path_model_trained is not None:
        model_from = tf.keras.models.load_model(path_model_trained,
                                                custom_objects={'DynamicUpsample': DynamicUpsample})
        # print("Model loaded\n", model_from.summary(expand_nested=True))
        # print("\n\n\nModel new\n", model_from.summary(expand_nested=True))
        for l, l_old in zip(model.layers[:-3], model_from.layers):
            # Print layer shapes
            # print("Layer: ", l.name, l.trainable, "new", [i.shape for i in l.get_weights()], "old", [i.shape for i in l_old.get_weights()])
            try:
                l.set_weights(l_old.get_weights())
            except Exception as e:
                print("Error in Layer: ", l.name, l.trainable, "new", [i.shape for i in l.get_weights()], "old",
                      [i.shape for i in l_old.get_weights()], e)
            if transfer_learning == "last": l.trainable = False
    mobilenetv3.trainable = trainable_backbone

    return model


if BACKBONE == "resnet50":
    model_name: str = ("Results/"
                       "Deeplab/"
                       "models/"
                       "deeplabv3plus_face_segmentation_pro_Aug_True_2024-04-27_16-04-39/"
                       "deeplabv3plus_face_segmentation_pro_Aug_True.h5")
else:
    model_name: str = ("Results/"
                       "Deeplab/"
                       "models/"
                       "deeplabv3plus_face_segmentation_pro_Aug_True_mobilenetv3_2024-04-28_23-31-01/"
                       "deeplabv3plus_face_segmentation_pro_Aug_True_mobilenetv3.h5")

# model_name=None
transfer_learning_type = "all"

if model_name is None:
    tag = "No_transfer_learning"
elif transfer_learning_type == "last":
    tag = "Transfer_learning_last"
else:
    tag = "Transfer_learning_all"

# model = tf.keras.models.load_model(model_name, custom_objects={'DynamicUpsample': DynamicUpsample})
if BACKBONE == "resnet50":
    model_nicco = DeeplabV3Plus_nicco(1, transfer_learning=transfer_learning_type, path_model_trained=model_name)
elif BACKBONE == "mobilenetv3":
    model_nicco = DeeplabV3Plus_nicco_mobilenetv3(1, transfer_learning=transfer_learning_type,
                                                  path_model_trained=model_name)
'''
for i, j in zip(model_nicco.get_weights(), model.get_weights()):
    print(i.shape, "|", j.shape,  "|", np.array_equal(i.shape, j.shape), "|", np.array_equal(i,j))

for l in model_nicco.layers:
    print(l.name, l.trainable, [i.shape for i in l.get_weights()])
'''
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS = 200
HIDDEN_UNITS = 400
EPOCHS_PATIENCE = 10
TRAINABLE_BASE = False
RESTORE_BEST_WEIGHTS = True
training_max_batches = None
use_pooling = False
use_validation_data_in_training = True
use_data_aug = False

real_train_path = home_path + "Dataset_merged/Train/Real/"
generated_train_path = home_path + "Dataset_merged/Train/Fake"
resize_dim = (IMG_WIDTH, IMG_HEIGHT)
model_input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

dataset_batch_size = BATCH_SIZE
train_batch_size = int(dataset_batch_size * 0.6)
validation_batch_size = int((dataset_batch_size - train_batch_size) * 0.6)
test_batch_size = dataset_batch_size - train_batch_size - validation_batch_size

parent_dir_train = home_path + "Dataset_merged/Train/"
real_dir_train = "Real"
sint_dir_train = "Fake"

parent_dir_val = home_path + "Dataset_merged/Valid/"
real_dir_val = "Real"
sint_dir_val = "Fake"

parent_dir_test = home_path + "Dataset_merged/Test/"
real_dir_test = "Real"
sint_dir_test = "Fake"

start_time = time.time()

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

# Balancing dataset
n_fakes = len(os.listdir(os.path.join(parent_dir_train, sint_dir_train)))
n_reals = len(os.listdir(os.path.join(parent_dir_train, real_dir_train)))
total = n_fakes + n_reals
weight_for_0 = (1 / n_reals) * total / 2.0
weight_for_1 = (1 / n_fakes) * total / 2.0

print("Real images: {}\nFake images: {}\nTotal: {}".format(n_reals, n_fakes, total))
class_weights = {0: weight_for_0, 1: weight_for_1}
print("Class weights: {}".format(class_weights))

model_nicco.summary()
model_nicco.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                    loss=keras.losses.BinaryCrossentropy(),
                    metrics=["accuracy",
                             keras.metrics.Precision(name='precision'),
                             keras.metrics.Recall(name='recall'),
                             keras.metrics.F1Score(name="F1")], weighted_metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=EPOCHS_PATIENCE,
                                               restore_best_weights=RESTORE_BEST_WEIGHTS,
                                               min_delta=0.001)

history = model_nicco.fit(train_dataset,
                          epochs=EPOCHS,
                          class_weight=class_weights,
                          callbacks=[early_stopping],
                          validation_data=val_dataset)

print(f"Training time: {str(timedelta(seconds=time.time() - start_time))}\n")
results_path = repo_path + "Results/Deeplab+fake_detection/"
model_folder = results_path + "trained_models/"

final_model_name = "deeplab+fake_detection_transfer_learning_" + str(tag) + "_" + BACKBONE + ".h5"
plot_eval_folder = results_path + "plots/" + final_model_name.split(".")[0]
performance_folder = results_path + "performance/" + final_model_name.split(".")[0] + "_" + time.strftime(
    "%Y-%m-%d_%H-%M-%S")
if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(plot_eval_folder):
    os.makedirs(plot_eval_folder)
if not os.path.exists(performance_folder):
    os.makedirs(performance_folder)

MDLNAME = model_folder + final_model_name

# Saving the model
model_nicco.save(MDLNAME)

best_epoch = np.argmin(history.history["val_loss"])

train_loss, train_accuracy, train_precision, train_recall, train_weighted_accuracy = model_nicco.evaluate(train_dataset)
val_loss, val_accuracy, val_precision, val_recall, val_weighted_accuracy = model_nicco.evaluate(val_dataset)
test_loss, test_accuracy, test_precision, test_recall, test_weighted_accuracy = model_nicco.evaluate(test_dataset)
train_f1 = 2 * (history.history["precision"][best_epoch] * history.history["recall"][best_epoch]) / (
        history.history["precision"][best_epoch] + history.history["recall"][best_epoch])
val_f1 = 2 * (history.history["val_precision"][best_epoch] * history.history["val_recall"][best_epoch]) / (
        history.history["val_precision"][best_epoch] + history.history["val_recall"][best_epoch])
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print("Performance:\n")
print("Best epoch: ", best_epoch)
print("Transfer learning: ", tag)
print("Training time: ", str(timedelta(seconds=time.time() - start_time)))
print("Training loss: ", train_loss)
print("Training accuracy: ", train_accuracy)
print("Training precision: ", train_precision)
print("Training recall: ", train_recall)
print("Training f1: ", train_f1)
print("Training weighted accuracy: ", train_weighted_accuracy)
print("Validation loss: ", val_loss)
print("Validation accuracy: ", val_accuracy)
print("Validation precision: ", val_precision)
print("Validation recall: ", val_recall)
print("Validation f1: ", val_f1)
print("Validation weighted accuracy: ", val_weighted_accuracy)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)
print("Test precision: ", test_precision)
print("Test recall: ", test_recall)
print("Test weighted accuracy: ", test_weighted_accuracy)
print("Test f1: ", test_f1)

# save evaluation plots
plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/training_loss.png")
plt.clf()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/training_accuracy.png")
plt.clf()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/validation_loss.png")
plt.clf()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/validation_accuracy.png")
plt.clf()

plt.plot(history.history["precision"])
plt.title("Training Precision")
plt.ylabel("precision")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/training_precision.png")
plt.clf()

plt.plot(history.history["recall"])
plt.title("Training Recall")
plt.ylabel("recall")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/training_recall.png")
plt.clf()

plt.plot(history.history["val_precision"])
plt.title("Validation Precision")
plt.ylabel("val_precision")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/validation_precision.png")
plt.clf()

plt.plot(history.history["val_recall"])
plt.title("Validation Recall")
plt.ylabel("val_recall")
plt.xlabel("epoch")
plt.savefig(plot_eval_folder + "/validation_recall.png")
plt.clf()

with open(performance_folder + "/performance.txt", "w") as f:
    f.write("Model: {}\n".format(MDLNAME))
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Transfer learning: {tag}\n")
    f.write("Training time: {}\n".format(str(timedelta(seconds=time.time() - start_time))))
    f.write("Training loss: {}\n".format(train_loss))
    f.write("Training accuracy: {}\n".format(train_accuracy))
    f.write("Training precision: {}\n".format(train_precision))
    f.write("Training recall: {}\n".format(train_recall))
    f.write("Training f1: {}\n".format(train_f1))
    f.write("Validation loss: {}\n".format(val_loss))
    f.write("Validation accuracy: {}\n".format(val_accuracy))
    f.write("Validation precision: {}\n".format(val_precision))
    f.write("Validation recall: {}\n".format(val_recall))
    f.write("Validation f1: {}\n".format(val_f1))
    f.write("Test loss: {}\n".format(test_loss))
    f.write("Test accuracy: {}\n".format(test_accuracy))
    f.write("Test precision: {}\n".format(test_precision))
    f.write("Test recall: {}\n".format(test_recall))
    f.write("Test f1: {}\n".format(test_f1))
