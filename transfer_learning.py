# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:34:14 2023

@author: ecero
"""

import os
import numpy as np
import argparse as ap
import json
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

def load_matrices(folder_path, input_label):
    reloaded_matrices = []
    filenames = os.listdir(folder_path)
    label_list = []
    for filename in tqdm(filenames, desc="Loaded"):
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

#making sure that base model runs in inference only mode (setting training = False), important for fine-tuning
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
base_model_encoding = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit output (binary classification)
dense1 = keras.layers.Dense(960, activation = "relu")
dense2 = keras.layers.Dense(864, activation = "relu")
dense3 = keras.layers.Dense(691, activation = "relu")
dense4 = keras.layers.Dense(483, activation = "relu")
dense5 = keras.layers.Dense(289, activation = "relu")

x = dense1(base_model_encoding)
x = dense2(x)
x = dense3(x)
x = dense4(x)
x = dense5(x)
outputs = keras.layers.Dense(1)(x)
#outputs = keras.layers.Dense(1)(base_model_encoding)

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
          epochs=run_params["top_model_epochs"], batch_size = run_params["batch_size"],shuffle = True)

# Unfreeze the base model
base_model.trainable = True

#recompiling the model to account for the change in trainability status of the base model
model.compile(optimizer=keras.optimizers.Adam(run_params["full_model_optimizer_learning_rate"]),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

print("Training the full model")
print(model.summary())
#TODO possibile applicare quella cosa del warm starting del paper?
#(https://arxiv.org/pdf/1910.08475.pdf On Warm-Starting Neural Network Training)


#Warm starting 
#defining lambda parameter (shrinking strength lambda must be in the range (0,1))
#paper got best results for lambda = 0.2
l = 0.6
#defining perturbation parameters
mu = 0
sigma = 0.01 #see paper figure 8 for reference to select lambda and sigma parameters
#definining number of repetitions
spr_reps = 2

#excluding input and global average pooling layers from shrink perturb repeat
excluded_layers = [0, 2]

#repeat
print("Training the full model with shrink-perturb-repeat for {} repetitions".format(spr_reps))
for r in range(spr_reps):
    print("Shrinking and perturbing each layer's weights")
    
    for i in range(len(model.layers)):
        if i not in excluded_layers:
            if i == 1:
                continue
                # layer = model.layers[i]
                # #print("Layer: {} Name {}".format(i, layer.name))
                # layer_weights = layer.get_weights()
                # for j in range(len(layer_weights)):
                #     for k in range(len(layer_weights[j])):
                #         if not isinstance(layer_weights[j][k], np.float32):
                #             layer_weights[j][k] = l*layer_weights[j][k] + np.random.normal(mu, sigma, size = layer_weights[j][k].shape)
                # #set mobilenet layer's new weights
                # layer.set_weights(layer_weights)
            else:
                #print("Layer: {} Name {}".format(i, model.layers[i].name))
                item = model.layers[i].get_weights()
                shrinked_w = l*item[0] + np.random.normal(loc = mu, scale = sigma, size = item[0].shape)
                shrinked_item = [shrinked_w, item[1]]
                model.layers[i].set_weights(shrinked_item)

    print("Fitting model")
    model.fit(x = training_cores, y= training_labels,
              epochs=run_params["full_model_epochs"], batch_size = run_params["batch_size"],shuffle = True) #TODO early stopping
    
    print("Evaluating model on training set for iteration {}".format(r+1))
    model.evaluate(x=training_cores, y=training_labels, batch_size = run_params["batch_size"])
#TODO salvare il modello

