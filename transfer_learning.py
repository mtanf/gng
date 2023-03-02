# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:34:14 2023

@author: ecero
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse as ap
import json
import tensorflow as tf
from tensorflow import keras
import h5py

def load_matrices(folder_path, input_label):
    reloaded_matrices = []
    filenames = os.listdir(folder_path)
    label_list = []
    for filename in filenames:
        reloaded_matrices.append(np.load(os.path.join(folder_path, filename), allow_pickle=False))
        if input_label == "Real":
            label_list.append(1)
        elif input_label == "Generated":
            label_list.append(0)
        else:
            print("Wrong label specified. Exiting")
            exit()

    return reloaded_matrices, label_list

parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required=True, help="Path to config json.")

args = vars(parser.parse_args())
json_path = args["path_to_json"]

with open(json_path) as f:
    run_params = json.load(f)
    
#reload core tensors of images
print("Loading training tensor cores")
real_cores, real_labels = load_matrices(run_params["real_compr"], "Real")
generated_cores, generated_labels = load_matrices(run_params["generated_compr"], "Generated")

training_cores = np.asarray([y for x in [real_cores, generated_cores] for y in x])
training_labels =np.asarray([y for x in [real_labels, generated_labels] for y in x])

#get input shape
input_shape = real_cores[0].shape

#instantiate a base model with pre-trained weight
base_model = keras.applications.MobileNetV3Large(input_shape = input_shape,
                                                 weights="imagenet",
                                                 include_top=False, #non include l'MLP con cui fanno prediction su imagenet
                                                 include_preprocessing=False) #preprocessing disabilitato ma la rete si aspetta tensori con valori nel range [0-255] #TODO forse possiamo non riscalare a mano l'HOSVD?

#freeze base model
base_model.trainable = False

#Create a new model on top of base model
inputs = keras.Input(shape=input_shape)

#making sure that base model runs in inference only mode, important for fine-tuning
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x) #TODO implementare un MLP vero
#defining model
model = keras.Model(inputs, outputs)
#compiling the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate = run_params["top_model_optimizer_learning_rate"]),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

print("Model summary")
print(model.summary())

print("Training the classifier on top of MobileNetV3")
model.fit(x = training_cores, y= training_labels,
          epochs=run_params["top_model_epochs"], batch_size = run_params["batch_size"])

# Unfreeze the base model
base_model.trainable = True

#recompiling the model to account for the change in trainability status of the base model
model.compile(optimizer=keras.optimizers.Adam(run_params["full_model_optimizer_learning_rate"]),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

print("Training the full model")
#TODO possibile applicare quella cosa del warm starting del paper?
#(https://arxiv.org/pdf/1910.08475.pdf On Warm-Starting Neural Network Training)

model.fit(x = training_cores, y= training_labels,
          epochs=run_params["full_model_epochs"], batch_size = run_params["batch_size"]) #TODO early stopping

model.evaluate(x=training_cores, y=training_labels, batch_size = run_params["batch_size"])
#TODO salvare il modello

