import os
import time
from glob import glob
import pickle
import keras
from sklearn.model_selection import train_test_split
from deeplab_mdl_def import DeeplabV3Plus, DeeplabV3Plus_mobilenet
from utils import *
from deeplab_mdl_def import DynamicUpsample
import matplotlib.pyplot as plt
from datetime import timedelta
from tensorflow.keras.metrics import MeanIoU
'''
import numpy as np
from PIL import Image
from scipy.io import loadmat
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''

IMAGE_SIZE = 448
BATCH_SIZE = 16
#NUM_CLASSES = 14
NUM_EPOCHS = 200
LR = 1e-4
EPOCHS_PATIENCE = 7
RESTORE_BEST_WEIGHTS = True
CALCULATE_CLASS_WEIGHTS = False
RELOAD_TRAINED_MODEL = True
AUGMENT_TRAIN_DATA = True
BACKBONE = "mobilenetv3" # "mobilenetv3" or "resnet50"

MACHINE = "apophis" # "apophis" or "mec-ai""
if MACHINE == "apophis":
    REPO_DIR= "/repo/tanfoni/"
else:
    REPO_DIR = "/homeRepo/tanfoni/"

DATA_DIR = REPO_DIR + "faceSegmentation/dataset_paid_integers"

MDLNAME = "deeplabv3plus_face_segmentation_pro_Aug_"+str(AUGMENT_TRAIN_DATA)+"_"+BACKBONE+".h5"
if RELOAD_TRAINED_MODEL:
    MDLNAME = ("Results/"
               "Deeplab/"
               "models/"
               "deeplabv3plus_face_segmentation_pro_Aug_True_2024-04-27_16-04-39/"
               "deeplabv3plus_face_segmentation_pro_Aug_True.h5")

VAL_IMG_FRAC = 0.2
TEST_IMG_FRAC = 0.1
SUBSET = False
SUBSET_SIZE = 200
COLORMAP_14 = {
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

COLORMAP_11 = {
    "background": [0, 0, 0],  # BGR
    "lips": [0, 0, 1],  # BGR
    "eyes": [0, 1, 0],  # BGR
    "nose": [1, 0, 0],  # BGR
    "face": [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],  # BGR
    "hair": [0, 1, 1],  # BGR
    "eyebrow": [1, 0, 1],  # BGR
    "ears": [1, 1, 0],  # BGR
    "teeth": [1, 1, 1],  # BGR
    "facial_hair": [0.7529411764705882, 0.7529411764705882, 1],  # BGR
    "glasses": [0.5019607843137255, 0.5019607843137255,0] #BGR
}

COLORMAP_TO_USE= COLORMAP_14 #TODO funzione per scegliere quale colormap usare

COLORMAP = {key: [color[2], color[1], color[0]] for key, color in COLORMAP_TO_USE.items()}
NUM_CLASSES=len(COLORMAP)

plot_eval_folder = "Results/Deeplab/plots/" + MDLNAME.split(".")[0] + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")
model_folder = "Results/Deeplab/models/" + MDLNAME.split(".")[0] + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")
performance_metrics_folder = "Results/Deeplab/performance_metrics/" + MDLNAME.split(".")[0] + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")


# # Check unique values in masks
# mask_folder = '/homeRepo/tanfoni/faceSegmentation/dataset_paid_integers/masks'
# unique_values = set()
# for mask_path in tqdm(os.listdir(mask_folder), desc="Counting unique values in masks"):
#     mask = np.array(Image.open(os.path.join(mask_folder, mask_path)))
#     unique_values.update(np.unique(mask))
# print("Unique label values:", unique_values)


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
all_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))
all_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))
# Subset if necessary
if SUBSET:
    all_images = all_images[:SUBSET_SIZE]
    all_masks = all_masks[:SUBSET_SIZE]

# Get number of train, val and test images
NUM_VAL_TRAIN_IMAGES = int((VAL_IMG_FRAC + TEST_IMG_FRAC) * len(all_images))
NUM_VAL_IMAGES = int(VAL_IMG_FRAC * len(all_images))
NUM_TEST_IMAGES = NUM_VAL_TRAIN_IMAGES - NUM_VAL_IMAGES
NUM_TRAIN_IMAGES = len(all_images) - NUM_VAL_TRAIN_IMAGES

train_images, val_images, train_masks, val_masks = train_test_split(all_images, all_masks,
                                                                    test_size=NUM_VAL_TRAIN_IMAGES, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(val_images, val_masks,
                                                                  test_size=NUM_TEST_IMAGES, random_state=42)

print("Train Images: {} | expected: {}".format(len(train_images), NUM_TRAIN_IMAGES))
print("Train Masks: {} | expected: {}".format(len(train_masks), NUM_TRAIN_IMAGES))
print("Val Images: {} | expected: {}".format(len(val_images), NUM_VAL_IMAGES))
print("Val Masks: {} | expected: {}".format(len(val_masks), NUM_VAL_IMAGES))
print("Test Images: {} | expected: {}".format(len(test_images), NUM_TEST_IMAGES))
print("Test Masks: {} | expected: {}".format(len(test_masks), NUM_TEST_IMAGES))

train_dataset = data_generator(train_images,train_masks, BATCH_SIZE, augment_data=AUGMENT_TRAIN_DATA)
val_dataset = data_generator(val_images, val_masks, BATCH_SIZE, augment_data=False)
test_dataset = data_generator(test_images, test_masks, BATCH_SIZE, augment_data=False)
test_dataset_no_resize = data_generator(test_images, test_masks, BATCH_SIZE, augment_data=False, resize_image=False)

if CALCULATE_CLASS_WEIGHTS or not os.path.exists('class_weights.pkl'):
    print("Calculating class weights...")
    class_weights = read_masks_and_compute_weights(os.path.join(DATA_DIR, "masks/"))
    with open('class_weights.pkl', 'wb') as pickle_file:
        pickle.dump(class_weights, pickle_file)
else:
    print("Reloading class weights pickle...")
    with open('class_weights.pkl', 'rb') as pickle_file:
        class_weights = pickle.load(pickle_file)

print("All done, class weights:")
print(class_weights)

iou = keras.metrics.MeanIoU(num_classes=NUM_CLASSES)
start_time = time.time()
if not RELOAD_TRAINED_MODEL or not os.path.exists(MDLNAME):
    print("Training model...")
    if BACKBONE == "mobilenetv3":
        model = DeeplabV3Plus_mobilenet(num_classes=NUM_CLASSES)
    else:
        model = DeeplabV3Plus(num_classes=NUM_CLASSES)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=loss,
        metrics=["accuracy"], weighted_metrics=["accuracy"]
    )

    print(model.summary())
    #Definining early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=EPOCHS_PATIENCE,
                                                   restore_best_weights=RESTORE_BEST_WEIGHTS, min_delta=0.05)

    #Fitting the model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS,
                        callbacks=[early_stopping],
                        class_weight=class_weights)

    print(f"Training took {str(timedelta(seconds=time.time() - start_time))}")

    if not os.path.exists(plot_eval_folder):
        os.makedirs(plot_eval_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)



    #Saving the model
    model.save(os.path.join(model_folder, MDLNAME))


    #Saving the training history
    with open(os.path.join(model_folder, MDLNAME.split(".")[0] + "_history.pkl"), "wb") as pickle_file:
        pickle.dump(history.history, pickle_file)
    #saving parameters
    with open(os.path.join(model_folder, MDLNAME.split(".")[0] + "_parameters.txt"), "w") as text_file:
        text_file.write(f"Image size: {IMAGE_SIZE}\n")
        text_file.write(f"Batch size: {BATCH_SIZE}\n")
        text_file.write(f"Number of classes: {NUM_CLASSES}\n")
        text_file.write(f"Number of epochs: {NUM_EPOCHS}\n")
        text_file.write(f"Learning rate: {LR}\n")
        text_file.write(f"Patience: {EPOCHS_PATIENCE}\n")
        text_file.write(f"Restore best weights: {RESTORE_BEST_WEIGHTS}\n")
        text_file.write(f"Calculate class weights: {CALCULATE_CLASS_WEIGHTS}\n")
        text_file.write(f"Reload trained model: {RELOAD_TRAINED_MODEL}\n")
        text_file.write(f"Data directory: {DATA_DIR}\n")
        text_file.write(f"Model name: {MDLNAME}\n")
        text_file.write(f"Validation image fraction: {VAL_IMG_FRAC}\n")
        text_file.write(f"Test image fraction: {TEST_IMG_FRAC}\n")
        text_file.write(f"Subset: {SUBSET}\n")
        text_file.write(f"Subset size: {SUBSET_SIZE}\n")
        text_file.write(f"Colormap: {COLORMAP}\n")
        text_file.write(f"Class weights: {class_weights}\n")
        text_file.write(f"Plot and evaluation folder: {plot_eval_folder}\n")
        text_file.write(f"Model folder: {model_folder}\n")
        text_file.write(f"Early stopping: {early_stopping}\n")
        text_file.write(f"History: {history}\n")
        text_file.write(f"Model: {model}\n")


    # Plot training loss
    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(plot_eval_folder + "/training_loss.png")
    # plt.show()
    plt.clf()


    # Plot training accuracy
    plt.plot(history.history["accuracy"])
    plt.title("Training Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig(plot_eval_folder + "/training_accuracy.png")
    # plt.show()
    plt.clf()


    # Plot validation loss
    plt.plot(history.history["val_loss"])
    plt.title("Validation Loss")
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    plt.savefig(plot_eval_folder + "/validation_loss.png")
    # plt.show()
    plt.clf()

    # Plot validation accuracy
    plt.plot(history.history["val_accuracy"])
    plt.title("Validation Accuracy")
    plt.ylabel("val_accuracy")
    plt.xlabel("epoch")
    plt.savefig(plot_eval_folder + "/validation_accuracy.png")
    # plt.show()
    plt.clf()

    # # Plot validation IoU
    # plt.plot(history.history["val_mean_io_u"])
    # plt.title("Validation IoU")
    # plt.ylabel("val_mean_io_u")
    # plt.xlabel("epoch")
    # plt.savefig(plot_eval_folder + "/validation_iou.png")
    # # plt.show()
    # plt.clf()

else:
    print("Reloading model...")
    #It needs to reload the custom DynamicUpsample layer as a custom_object
    model = keras.models.load_model(MDLNAME, custom_objects={'DynamicUpsample': DynamicUpsample})
    print(model.summary())

# Saving main performance metrics
# train_loss, train_accuracy, train_weighted_accuracy = model.evaluate(train_dataset)
# val_loss, val_accuracy, val_weighted_accuracy = model.evaluate(val_dataset)
# test for different image sizes

print("Evaluating model on images with different sizes")
test_loss, test_accuracy, test_weighted_accuracy = model.evaluate(test_dataset_no_resize)
print(f"Test loss (No resize): {test_loss}, Test accuracy: {test_accuracy}")
 #TODO problemi con mutiny pro



print("Evaluating model on images with uniform size (resize)")
test_loss, test_accuracy, test_weighted_accuracy = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test weighted accuracy: {test_weighted_accuracy}")

# save test results
with open(os.path.join(model_folder, MDLNAME.split(".")[0] + "_test_results.txt"), "w") as text_file:
    text_file.write(f"Test loss (No resize): {test_loss}, Test accuracy: {test_accuracy}\n")
    text_file.write(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}\n")
    # text_file.write(f"Test IoU: {test_iou}\n")
# save train and validation performance metrics for each epoch
with open(os.path.join(model_folder, MDLNAME.split(".")[0] + "_history.pkl"), "rb") as pickle_file:
    history = pickle.load(pickle_file)
    with open(os.path.join(model_folder, MDLNAME.split(".")[0] + "_performance_metrics.txt"), "w") as text_file:
        text_file.write(f"Epoch | Train loss | Train accuracy | Val loss | Val accuracy\n")
        for i in range(len(history["loss"])):
            text_file.write(f"{i} | {history['loss'][i]} | {history['accuracy'][i]} | {history['val_loss'][i]} | {history['val_accuracy'][i]}\n")
        text_file.write(f"\n\n\nBest epoch: {np.argmin(history['val_loss'])},"
                        f" with train accuracy: {history['accuracy'][np.argmin(history['val_loss'])]}"
                        f" and validation accuracy: {history['val_accuracy'][np.argmin(history['val_loss'])]}\n")

        text_file.write(f"\n\nTest loss (No resize): {test_loss}, Test accuracy: {test_accuracy}\n")
        text_file.write(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}\n")
        text_file.write(f"\n\nTraining took {str(timedelta(seconds=time.time() - start_time))}\n")

