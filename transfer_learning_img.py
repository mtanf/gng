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
    #x = dense2(x)
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

# Definisci la funzione per caricare e decodificare le immagini
def load_and_decode_image(filename, label, image_size):
    # Load and decode the image from the file
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    # Resize the image to the desired size
    image = tf.image.resize(image, image_size)
    # Convert label to a tensor
    label = tf.convert_to_tensor(label)
    return image, label

parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required=True, help="Path to config json.")

args = vars(parser.parse_args())
json_path = args["path_to_json"]

with open(json_path) as f:
    run_params = json.load(f)
    
#reload core tensors of images
print("Loading training images")

# Definisci le cartelle contenenti le immagini
real_train_path = run_params["real_imgs_train"]
generated_train_path = run_params["generated_imgs_train"]
# Dimensione resize immagini
resize_dim = (run_params["new_img_dim"], run_params["new_img_dim"])
model_input_shape =(run_params["new_img_dim"], run_params["new_img_dim"],3)


if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")
    
# Definisci il batch size
dataset_batch_size = 512


parent_dir_train = run_params["dataset_dir_train"]
real_dir_train = run_params["real_imgs_train"]
sint_dir_train = run_params["generated_imgs_train"]

# Create a training dataset from the two directories
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_train,
    labels="inferred",
    class_names = [real_dir_train, sint_dir_train],
    label_mode='int',
    color_mode='rgb',
    batch_size=dataset_batch_size,
    image_size=resize_dim,
    shuffle=True,
    seed=123
)

parent_dir_val = run_params["dataset_dir_val"]
real_dir_val = run_params["real_imgs_val"]
sint_dir_val = run_params["generated_imgs_val"]

val_dataset  =  tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_val,
    labels="inferred",
    class_names = [real_dir_val, sint_dir_val],
    label_mode='int',
    color_mode='rgb',
    batch_size=dataset_batch_size,
    image_size=resize_dim,
    shuffle=True,
    seed=123
)

parent_dir_test = run_params["dataset_dir_test"]
real_dir_test = run_params["real_imgs_test"]
sint_dir_test = run_params["generated_imgs_test"]

test_dataset  =  tf.keras.preprocessing.image_dataset_from_directory(
    parent_dir_test,
    labels="inferred",
    class_names = [real_dir_test, sint_dir_test],
    label_mode='int',
    color_mode='rgb',
    batch_size=dataset_batch_size,
    image_size=resize_dim,
    shuffle=True,
    seed=123
)

reload_base = True
if reload_base:
    model = get_compiled_model(model_input_shape, trainable_base = False, lr = run_params["top_model_optimizer_learning_rate"])
    model.load_weights(os.path.join("checkpoints", "MobileNetFrozen_imgs_weights"))
    
else:
    #get frozen model
    model = get_compiled_model(model_input_shape, trainable_base = False, lr = run_params["top_model_optimizer_learning_rate"])
    print("Model summary")
    print(model.summary())
    print("Training the classifier on top of MobileNetV3")
    # Train the model in batches 
    idx = 1
    for images, labels in train_dataset:
        print("Training batch {} out of {}".format(idx, len(train_dataset)))
        model.fit(x = images, y = labels, epochs=run_params["top_model_epochs"],shuffle = True)
        idx +=1
        
    model.save_weights(os.path.join("checkpoints", "MobileNetFrozen_imgs_weights"))

idx = 1
chunk_test_accuracies = []
for test_images, test_labels in test_dataset:
    print("Test batch {} out of {}".format(idx, len(test_dataset)))
    chunk_test_accuracies.append(model.evaluate(test_images,test_labels))
    idx +=1

for i in range(len(chunk_test_accuracies)):
    print("Chunk {} accuracy: {}".format(i+1,chunk_test_accuracies[i]))
    
a


del model
keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


#get unfrozen model
model = get_compiled_model(model_input_shape, trainable_base = True, lr = run_params["full_model_optimizer_learning_rate"])
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
        model = get_compiled_model(model_input_shape, trainable_base = True, lr = run_params["full_model_optimizer_learning_rate"])
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
    model.fit(train_dataset, epochs=run_params["top_model_epochs"],shuffle = True)
    
    print("Evaluating model on training set for iteration {}".format(r+1))
    model.evaluate(train_dataset)
        
    model.save_weights(os.path.join("checkpoints", "SPR_imgs_run_{}_weights".format(r+1)))
    del model
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    
    
#TODO salvare il modello

