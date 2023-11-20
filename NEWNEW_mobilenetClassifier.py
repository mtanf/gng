import os
import numpy as np
import argparse as ap
import json
import tensorflow as tf
from tensorflow import keras
import gc
import shap
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

print("SHAP Version : {}".format(shap.__version__))


def get_encoder(input_shape, trainable_last_layer=False):
    base_model = keras.applications.MobileNetV3Large(include_top=False, weights="imagenet", input_shape=(200,200,3))

    for layer in base_model.layers[:-1]:
        layer.trainable = False

    if trainable_last_layer:
        base_model.layers[-1].trainable = True

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    encoder_output = keras.layers.Flatten()(x)

    encoder_model = keras.Model(inputs, encoder_output)
    return encoder_model


def get_classifier(encoder, num_classes=1):
    inputs = encoder.input
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(encoder.output)
    classifier_model = keras.Model(inputs, outputs)
    return classifier_model


def build_model(input_shape, trainable_base=False, lr=1e-3, use_pooling=False):
    # !!!!!!
    # MobileNetV3 models expect their inputs to be float tensors of pixels with values in the [0-255] range
    # !!!!!!
    # instantiate a base model with pre-trained weight
    base_model = keras.applications.MobileNetV3Large(input_shape=input_shape,
                                                     weights="imagenet",
                                                     include_top=False,
                                                     # non include l'MLP con cui fanno prediction su imagenet
                                                     include_preprocessing=True)  # preprocessing disabilitato ma la rete si aspetta tensori con valori nel range [0-255] #TODO forse possiamo non riscalare a mano l'HOSVD?

    if trainable_base:
        # unfreeze base model
        base_model.trainable = True
    else:
        # freeze the base model
        base_model.trainable = False

    # Create a new model on top of base model
    inputs = keras.Input(shape=input_shape)

    # making sure that base model runs in inference only mode (setting training = False), important for fine-tuning
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    if use_pooling:
        base_model_encoding = keras.layers.GlobalAveragePooling2D()(x)
        classification_layer = keras.layers.Dense(1000, activation="relu")(base_model_encoding)
        outputs = keras.layers.Dense(1, activation="sigmoid")(classification_layer)
    else:
        # Flatten the features
        x = keras.layers.Flatten()(x)
        classification_layer = keras.layers.Dense(100, activation="relu")(x)
        outputs = keras.layers.Dense(1, activation="sigmoid")(classification_layer)

    # defining model
    model = keras.Model(inputs, outputs)
    # # compiling the model
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
    #               loss=keras.losses.BinaryCrossentropy(from_logits=False),
    #               metrics=[keras.metrics.BinaryAccuracy()])
    return model



def load_and_preprocess_image(filename, label, image_size):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, image_size)
    # image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    label = tf.convert_to_tensor(label)
    return image, label


parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required=True, help="Path to config json.")
args = vars(parser.parse_args())
json_path = args["path_to_json"]

training_max_batches = None
use_pooling = False
use_validation_data_in_training = True
use_data_aug = False

main_model_output_path = "trained_classifiers"
classifier_name = "MobileNetV3_frozen_useDataAug_" + str(use_data_aug)
weight_output_path = os.path.join(main_model_output_path, classifier_name + "_weights")
performance_log_output_path = os.path.join(main_model_output_path, classifier_name + "_performance")
evaluate_trained_model = True

main_shap_output_path = "SHAPOut"
shap_output_path = os.path.join(main_shap_output_path, classifier_name + "_shap_explanations")

if not os.path.exists(main_model_output_path):
    os.mkdir(main_model_output_path)

if not os.path.exists(weight_output_path):
    os.mkdir(weight_output_path)

if not os.path.isdir(main_shap_output_path):
    os.mkdir(main_shap_output_path)

if not os.path.isdir(shap_output_path):
    os.mkdir(shap_output_path)

with open(json_path) as f:
    run_params = json.load(f)

real_train_path = run_params["real_imgs_train"]
generated_train_path = run_params["generated_imgs_train"]
resize_dim = (run_params["new_img_dim"], run_params["new_img_dim"])
model_input_shape = (run_params["new_img_dim"], run_params["new_img_dim"], 3)

dataset_batch_size = run_params["batch_size"]
train_batch_size = int(dataset_batch_size * 0.6)
validation_batch_size = int((dataset_batch_size - train_batch_size) * 0.6)
test_batch_size = dataset_batch_size - train_batch_size - validation_batch_size

parent_dir_train = run_params["dataset_dir_train"]
real_dir_train = run_params["real_imgs_train"]
sint_dir_train = run_params["generated_imgs_train"]

parent_dir_val = run_params["dataset_dir_val"]
real_dir_val = run_params["real_imgs_val"]
sint_dir_val = run_params["generated_imgs_val"]

parent_dir_test = run_params["dataset_dir_test"]
real_dir_test = run_params["real_imgs_test"]
sint_dir_test = run_params["generated_imgs_test"]



print("Loading training images")
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_train,
    labels="inferred",
    class_names=[real_dir_train, sint_dir_train],
    label_mode='int',
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
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_dataset_aug = data_augmentation.flow_from_directory(
        parent_dir_train,
        target_size=resize_dim,
        batch_size=train_batch_size,
        class_mode='binary',
        shuffle=True,
        seed=123
    )


print("Loading validation images")
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_val,
    labels="inferred",
    class_names=[real_dir_val, sint_dir_val],
    label_mode='int',
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
    label_mode='int',
    color_mode='rgb',
    batch_size=test_batch_size,
    image_size=resize_dim,
    shuffle=True,
    seed=123
)

model = build_model(model_input_shape)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=run_params["top_model_optimizer_learning_rate"]),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy()])

if use_data_aug:
    if use_validation_data_in_training:
        model.fit(train_dataset_aug, epochs=run_params["full_model_epochs"], validation_data=val_dataset)
    else:
        model.fit(train_dataset_aug, epochs=run_params["full_model_epochs"])
else:
    if use_validation_data_in_training:
        model.fit(train_dataset, epochs=run_params["full_model_epochs"], validation_data=val_dataset)
    else:
        model.fit(train_dataset, epochs=run_params["full_model_epochs"])



# if use_data_aug:
#     for epoch in range(run_params["training_epochs"]):
#         print("Epoch {} out of {}".format(epoch + 1, run_params["training_epochs"]))
#         for idx, (images, labels) in enumerate(train_dataset_aug):
#             print("Training batch {} out of {}".format(idx + 1, len(train_dataset_aug)))
#             if use_validation_data_in_training:
#                 model.fit(x=images, y=labels, epochs=run_params["NoFineTuning_BatchEpochs"], shuffle=True,
#                           validation_data=val_dataset)
#             else:
#                 model.fit(x=images, y=labels, epochs=run_params["NoFineTuning_BatchEpochs"], shuffle=True)
#             if training_max_batches is not None:
#                 if idx >= training_max_batches:
#                     print("Breaking, max training batches reached ({})".format(training_max_batches))
#                     break
#             else:
#                 if idx >= len(train_dataset_aug):
#                     # print("Breaking, max training batches reached ({})".format(len(train_dataset_aug)))
#                     break
# else:
#     for epoch in range(run_params["training_epochs"]):
#         print("Epoch {} out of {}".format(epoch + 1, run_params["training_epochs"]))
#         for idx, (images, labels) in enumerate(train_dataset):
#             print("Training batch {} out of {}".format(idx + 1, len(train_dataset)))
#             if use_validation_data_in_training:
#                 model.fit(x=images, y=labels, epochs=run_params["NoFineTuning_BatchEpochs"], shuffle=True,
#                           validation_data=val_dataset)
#             else:
#                 model.fit(x=images, y=labels, epochs=run_params["NoFineTuning_BatchEpochs"], shuffle=True)
#             if training_max_batches is not None:
#                 if idx >= training_max_batches:
#                     print("Breaking, max training batches reached ({})".format(training_max_batches))
#                     break
#             else:
#                 if idx >= len(train_dataset):
#                     # print("Breaking, max training batches reached ({})".format(len(train_dataset_aug)))
#                     break

print("Saving model")
model.save(os.path.join(weight_output_path, classifier_name + "_weights.h5"))

del model
K.clear_session()
gc.collect()
model = keras.models.load_model(os.path.join(weight_output_path, classifier_name + "_weights.h5"))

if evaluate_trained_model:
    print("Evaluating performance")
    if not os.path.exists(performance_log_output_path):
        os.mkdir(performance_log_output_path)

    train_set_eval = model.evaluate(train_dataset, batch_size=test_batch_size)
    val_set_eval = model.evaluate(val_dataset, batch_size=test_batch_size)
    test_set_eval = model.evaluate(test_dataset, batch_size=test_batch_size)

    train_set_eval = np.array(train_set_eval)
    val_set_eval = np.array(val_set_eval)
    test_set_eval = np.array(test_set_eval)

    # evaluating F1 score
    train_set_eval = np.append(train_set_eval, 2 * train_set_eval[1] * train_set_eval[2] / (train_set_eval[1] + train_set_eval[2]))
    val_set_eval = np.append(val_set_eval, 2 * val_set_eval[1] * val_set_eval[2] / (val_set_eval[1] + val_set_eval[2]))
    test_set_eval = np.append(test_set_eval, 2 * test_set_eval[1] * test_set_eval[2] / (test_set_eval[1] + test_set_eval[2]))

    # evaluating precision
    train_set_eval = np.append(train_set_eval, 2 * train_set_eval[1] * train_set_eval[3] / (train_set_eval[1] + train_set_eval[3]))
    val_set_eval = np.append(val_set_eval, 2 * val_set_eval[1] * val_set_eval[3] / (val_set_eval[1] + val_set_eval[3]))
    test_set_eval = np.append(test_set_eval, 2 * test_set_eval[1] * test_set_eval[3] / (test_set_eval[1] + test_set_eval[3]))

    # evaluating recall
    train_set_eval = np.append(train_set_eval, 2 * train_set_eval[1] * train_set_eval[4] / (train_set_eval[1] + train_set_eval[4]))
    val_set_eval = np.append(val_set_eval, 2 * val_set_eval[1] * val_set_eval[4] / (val_set_eval[1] + val_set_eval[4]))
    test_set_eval = np.append(test_set_eval, 2 * test_set_eval[1] * test_set_eval[4] / (test_set_eval[1] + test_set_eval[4]))



    with open(os.path.join(performance_log_output_path, classifier_name + "_performance_log.txt"), "w") as f:
        f.write("Performance log for model {}\n".format(classifier_name))
        f.write("Train set: {}\nValidation set: {}\nTest set: {}\n".format(parent_dir_train, parent_dir_val,
                                                                           parent_dir_test))
        f.write("Train set Accuracy:{:.2f}%\tLoss:{:.4f}\n".format(train_set_eval[1] * 100, train_set_eval[0]))
        f.write("Validation set Accuracy:{:.2f}%\tLoss:{:.4f}\n".format(val_set_eval[1] * 100, val_set_eval[0]))
        f.write("Train set Accuracy:{:.2f}%\tLoss:{:.4f}\n".format(test_set_eval[1] * 100, test_set_eval[0]))

        f.write("Train set F1 score:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}\n".format(train_set_eval[5], train_set_eval[6], train_set_eval[7]))
        f.write("Validation set F1 score:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}\n".format(val_set_eval[5], val_set_eval[6], val_set_eval[7]))
        f.write("Test set F1 score:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}\n".format(test_set_eval[5], test_set_eval[6], test_set_eval[7]))

    f.close()

num_images_shap = 10
shap_data_origin = "val"

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

shap_values = explainer(X_shap[:num_images_shap], outputs=shap.Explanation.argsort.flip[:num_images_shap], max_evals=1000)

mapping = dict(zip([0, 1], class_names))

print("SHAP results")
print("Actual Labels    : {}".format([mapping[i] for i in Y_shap[:num_images_shap]]))
preds_shap = model.predict(X_shap[:num_images_shap])
pred_labels_shap = np.where(preds_shap > 0.5, 1, 0).transpose().tolist()[0]

print("Predicted Labels : {}".format([mapping[i] for i in pred_labels_shap]))
print("Sigmoid activation values : {}".format(preds_shap))

predictions = model.predict(X_shap[:num_images_shap])
predictions_labels = np.where(predictions > 0.5, 1, 0)

predictions_to_plot = []
for i in range(num_images_shap):
    predictions_to_plot.append(
        "Pred:{} | True:{}".format(mapping[predictions_labels.transpose().tolist()[0][i]], mapping[Y_shap.tolist()[i]]))
predictions_to_plot = np.array([[i] for i in predictions_to_plot])

shap.image_plot(shap_values, pixel_values=X_shap_print[:num_images_shap], labels=predictions_to_plot, show=False)
plt.savefig(os.path.join(shap_output_path, "Shap_results_{}_set.png".format(shap_data_origin)))
matplotlib.pyplot
plt.show()