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
from tensorflow.keras import metrics
from sklearn.utils.class_weight import compute_class_weight
from deeplab_mdl_def import DeeplabV3Plus
from deeplab_mdl_def import DynamicUpsample

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def create_mlp(input_dim, neurons):
    """
    Create a multi-layer perceptron (MLP) model based on a list of neuron counts for each hidden layer.

    Parameters:
    - neurons (list of int): A list where each element specifies the number of neurons in a corresponding hidden layer.

    Returns:
    - A Keras model instance with the specified architecture.
    """
    mlp = keras.models.Sequential()
    # Add the input layer explicitly if required or the first hidden layer can act as the input layer
    mlp.add(keras.layers.InputLayer(input_shape=(input_dim,)))

    # Add each hidden layer specified in the neurons list
    for i, num_neurons in enumerate(neurons):
        mlp.add(keras.layers.Dense(num_neurons, activation='relu', name=f'hidden_layer_{i + 1}'))

    # Add the output layer
    mlp.add(keras.layers.Dense(1, activation='sigmoid',
                               name='output_layer'))
    '''
    # Compile the model (optional here, could be done outside this function)
    mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    '''
    return mlp


def classifier(input_shape, segm, segm_n_classes=11, compression_units=200,
               mlp_neurons=[20, 20, 20], classifier_n_classes=2,
               trainable_base=False, lr=1e-3, use_pooling=False, hidden_units=200):
    encoder = tf.keras.applications.MobileNetV3Large(input_shape=input_shape,
                                                     weights="imagenet",
                                                     include_top=False,
                                                     include_preprocessing=True)
    compression_layer = tf.keras.layers.Dense(units=compression_units, activation="sigmoid", name="compression_layer")
    classifier_mlp = create_mlp(input_dim=compression_units,
                                neurons=mlp_neurons)
    final_classification_layer = tf.keras.layers.Dense(units=1, activation="sigmoid", name="final_classification_layer")

    # set the encoder to trainable or not
    encoder.trainable = trainable_base
    # set the segmenation model to not trainable
    segm.trainable = False

    # Input layer -> image
    inputs = keras.Input(shape=input_shape)
    # Extract face masks with segm
    face_masks = segm(inputs)

    face_masks = tf.argmax(face_masks, axis=-1)

    # Extract patches from the original image according to each class
    expanded_face_masks = tf.expand_dims(face_masks, axis=-1)  # Expand to [batch_size, height, width, 1]
    # Initialize a list of lists to hold the patches for each class
    class_patches = [[] for _ in range(segm_n_classes)]

    # Extract patches for each class
    for class_index in range(segm_n_classes):
        # Create the condition tensor, automatically broadcasted across the channel dimension
        condition = tf.equal(expanded_face_masks, class_index)
        # Extract patches where condition is True, otherwise fill with zeros
        patches_for_class = tf.where(condition, inputs, tf.zeros_like(inputs))
        # Append these patches to the corresponding class list
        class_patches[class_index].append(patches_for_class)

    encodings = []
    final_predictions = []
    for patches in class_patches:
        if patches:  # Ensure the list is not empty
            patches_tensor = patches[0]  # Assuming each class has one tensor of patches
            # Call the model on the tensor directly
            class_predictions = encoder(patches_tensor)
            # Apply Global Average Pooling to the output of the model
            class_predictions = tf.keras.layers.GlobalAveragePooling2D()(class_predictions)
            # Apply the compression layer
            class_predictions = compression_layer(class_predictions)
            # Store or process the predictions as needed
            encodings.append(class_predictions)
            # Pass each encoding through the MLP to obtain predictions
            mlp_predictions = classifier_mlp(class_predictions)
            # Store or process the MLP predictions as needed
            final_predictions.append(mlp_predictions)

    # Apply final dense layer for classification
    final_predictions = tf.convert_to_tensor(final_predictions)
    outputs = final_classification_layer(final_predictions)

    # defining model
    model = tf.keras.Model(inputs, outputs)

    # compiling the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(),
                      metrics.Precision(),
                      metrics.Recall(),
                      metrics.F1Score(average="weighted")
                  ])
    print(model.summary())
    return model


# Test code
input_tensor = np.random.rand(2, 448, 448, 3).astype('float32')

# Reload trained face segmentator
seg_name = "deeplabv3plus_face_segmentation_augmentation_class_weights_latest_fixConv_different_rate"
segm = keras.models.load_model(seg_name + ".h5", custom_objects={'DynamicUpsample': DynamicUpsample})

model = classifier(
    input_shape=(448, 448, 3),
    segm=segm,
    segm_n_classes=11,
    compression_units=200,
    mlp_neurons=[20, 20, 20],
    classifier_n_classes=2,
    trainable_base=False,
    lr=1e-3,
    use_pooling=False
)
