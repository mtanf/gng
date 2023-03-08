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
import gc
import cv2

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


def get_compiled_model(input_shape, trainable_base = False, lr = 1e-3):
    #instantiate a base model with pre-trained weight
    base_model = keras.applications.MobileNetV3Large(input_shape = input_shape,
                                                     weights="imagenet",
                                                     include_top=False, #non include l'MLP con cui fanno prediction su imagenet
                                                     include_preprocessing=True) #preprocessing disabilitato ma la rete si aspetta tensori con valori nel range [0-255] #TODO forse possiamo non riscalare a mano l'HOSVD?

    if trainable_base:
        #unfreeze base model
        base_model.trainable = True
    else:
        #freeze the base model
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
    # x = dense3(x)
    # x = dense4(x)
    # x = dense5(x)
    outputs = keras.layers.Dense(1)(x)
    #outputs = keras.layers.Dense(1)(base_model_encoding)

    #defining model
    model = keras.Model(inputs, outputs)
    #compiling the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = lr),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])
    return model

def load_imgs(folder_path, input_label, new_img_dim = 448):
    loaded_imgs = []
    filenames = os.listdir(folder_path)
    label_list = []
    for filename in tqdm(filenames, desc="Loaded"):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, (new_img_dim, new_img_dim))
        loaded_imgs.append(img)
        if input_label == "Real":
            label_list.append(1)
        elif input_label == "Generated":
            label_list.append(0)
        else:
            print("Wrong label specified. Exiting")
            exit()

    return loaded_imgs, label_list

parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required=True, help="Path to config json.")

args = vars(parser.parse_args())
json_path = args["path_to_json"]

with open(json_path) as f:
    run_params = json.load(f)
    
#reload core tensors of images
print("Loading training images")
new_img_dim = run_params["new_img_dim"]
real_imgs_train, real_labels_train = load_imgs(run_params["real_imgs_train"], "Real", new_img_dim = new_img_dim)
generated_imgs_train, generated_labels_train = load_imgs(run_params["generated_imgs_train"], "Generated", new_img_dim = new_img_dim)

training_imgs = np.asarray([y for x in [real_imgs_train, generated_imgs_train] for y in x])
training_labels =np.asarray([y for x in [real_labels_train, generated_labels_train] for y in x])

a
# print("Loading validation images")
# real_imgs_val, real_labels_val = load_imgs(run_params["real_imgs_val"], "Real", new_img_dim = new_img_dim)
# generated_imgs_val, generated_labels_val = load_imgs(run_params["generated_imgs_val"], "Generated", new_img_dim = new_img_dim)

# training_imgs_val = np.asarray([y for x in [real_imgs_val, generated_imgs_val] for y in x])
# training_labels_val =np.asarray([y for x in [real_labels_val, generated_labels_val] for y in x])


# print("Loading test images")
# real_imgs_val, real_labels_val = load_imgs(run_params["real_imgs_val"], "Real", new_img_dim = new_img_dim)
# generated_imgs_val, generated_labels_val = load_imgs(run_params["generated_imgs_val"], "Generated", new_img_dim = new_img_dim)

training_imgs_val = np.asarray([y for x in [real_imgs_val, generated_imgs_val] for y in x])
training_labels_val =np.asarray([y for x in [real_labels_val, generated_labels_val] for y in x])

#get input shape
input_shape = (new_img_dim, new_img_dim, 3)

#get frozen model
model = get_compiled_model(input_shape, trainable_base = False, lr = run_params["top_model_optimizer_learning_rate"])

print("Model summary")
print(model.summary())

if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")

print("Training the classifier on top of MobileNetV3")
model.fit(x = training_imgs, y= training_labels,
          epochs=run_params["top_model_epochs"], batch_size = run_params["batch_size"],shuffle = True)


model.save_weights(os.path.join("checkpoints", "MobileNetFrozen_imgs_weights"))
del model
keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


#get unfrozen model
model = get_compiled_model(input_shape, trainable_base = True, lr = run_params["full_model_optimizer_learning_rate"])
model.load_weights(os.path.join("checkpoints", "MobileNetFrozen_imgs_weights"))

print("Training the full model")
print(model.summary())
#TODO possibile applicare quella cosa del warm starting del paper?
#(https://arxiv.org/pdf/1910.08475.pdf On Warm-Starting Neural Network Training)


#Warm starting 
#defining lambda parameter (shrinking strength lambda must be in the range (0,1))
#paper got best results for lambda = 0.2
l = 0.8
#defining perturbation parameters
mu = 0
sigma = 0.05
noise_scale = 0.0001 #see paper figure 8 for reference to select lambda and sigma parameters
#definining number of repetitions
spr_reps = run_params["spr_reps"]

#excluding input and global average pooling layers from shrink perturb repeat
excluded_layers = [0, 2]

#repeat
print("Training the full model with shrink-perturb-repeat for {} repetitions".format(spr_reps))
#TODO questo andrebbe fatto quando ci sono più batch di immagini su cui addestrarsi
#non ripetendo l'addestramento N volte sullo stesso batch
#appena avremo più immagini va modificato
for r in range(spr_reps):
    print("Shrinking and perturbing each layer's weights")
    
    if r>0:
        model = get_compiled_model(input_shape, trainable_base = True, lr = run_params["full_model_optimizer_learning_rate"])
        model.load_weights(os.path.join("checkpoints", "SPR_imgs_run_{}_weights".format(r)))
        
    for i in range(len(model.layers)):
        if i not in excluded_layers:
            if i != 1:
                #print("Layer: {} Name {}".format(i, model.layers[i].name))
                # item = model.layers[i].get_weights()
                # shrinked_w = l*item[0] + noise_scale*np.random.normal(loc = mu, scale = sigma, size = item[0].shape)
                # shrinked_item = [shrinked_w, item[1]]
                # model.layers[i].set_weights(shrinked_item)
                continue

    print("Fitting model")
    model.fit(x = training_imgs, y= training_labels,
              epochs=run_params["full_model_epochs"], batch_size = run_params["batch_size"],shuffle = True) #TODO early stopping
    
    print("Evaluating model on training set for iteration {}".format(r+1))
    model.evaluate(x=training_imgs, y=training_labels, batch_size = run_params["batch_size"])
        
    model.save_weights(os.path.join("checkpoints", "SPR_imgs_run_{}_weights".format(r+1)))
    del model
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    
    
#TODO salvare il modello

