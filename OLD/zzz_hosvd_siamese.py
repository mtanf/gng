# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:39:38 2023

@author: ecero
"""


import numpy as np
import cv2
import os
import tensorly as tl

import matplotlib.pyplot as plt
import random
import argparse as ap
import json
import pickle
import shutil

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Nadam

from keras.layers import BatchNormalization, Concatenate
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPool2D, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.utils import normalize

from sklearn.metrics import confusion_matrix
from  sklearn.preprocessing import MinMaxScaler


def load_images_from_folder(folder, new_img_dim):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        #load
        img = cv2.imread(os.path.join(folder,filename))
        #resize
        img = cv2.resize(img, (new_img_dim, new_img_dim))
        # #show
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if img is not None:
            images.append(img)
    return images, filenames

def tucker_decomposed_imgs(img_list, tucker_rank):
    core_imgs = []
    for img in img_list:
        core, factors = tl.decomposition.tucker(img, rank=tucker_rank)
        core_imgs.append(core)
    return core_imgs

def rescale_cores(core_list):
    rescaled_cores = []
    for item in core_list:
        scaled = normalize(item)
        rescaled_cores.append(scaled)
    return rescaled_cores

def save_matrices(matrices_list, original_filenames, savepath):
    #saves each matrix in a matrix list to a .npy file
    for i in range(len(matrices_list)):
        item = matrices_list[i]
        orig_name = original_filenames[i]
        #remove original file extension from filename
        orig_name = orig_name.split(".")[0]
        new_name = orig_name + "_HOSVD_Core"
        np.save(os.path.join(savepath, new_name), item, allow_pickle = False)
    return

def load_matrices(folder_path):
    reloaded_matrices = []
    filenames = os.listdir(folder_path)
    for filename in filenames:
        reloaded_matrices.append(np.load(os.path.join(folder_path, filename), allow_pickle = False))
        
    return reloaded_matrices

def siamese_input_gen(l_input_all, r_input_all, label_all, n=1): 
    
    while True:
        l = len(l_input_all)
        
        for ndx in range(0, l, n):
            
            l_input = l_input_all[ndx:min(ndx + n, l)]
            r_input = r_input_all[ndx:min(ndx + n, l)]
            labels = label_all[ndx:min(ndx + n, l)]
            yield ([l_input, r_input], labels)

def coupler(positive_imgs, negative_imgs, template_imgs):
    num_template = len(template_imgs)
    
    same_class_imgs = []
    different_class_imgs = []
    max_coupleable = num_template  #TODO set this as parameters for function
   
    for i in range(len(positive_imgs) - 1): #
        if i <= max_coupleable:
            sample = positive_imgs[i]
            remaining = positive_imgs[i:]
            
            for item in remaining:
                couple = []
                couple.append(sample)
                couple.append(item)
                same_class_imgs.append(couple)
            for item in negative_imgs:
                couple = []
                couple.append(sample)
                couple.append(item)
                different_class_imgs.append(couple)
    print("Number of same class image pairs = {}, Number of different class image pairs = {}, total sample pairs: {}\n".format(len(same_class_imgs), len(different_class_imgs), len(same_class_imgs) + len(different_class_imgs) ))
    
    return same_class_imgs, different_class_imgs

def siamese_input_creator(positive_cores, negative_cores, template_cores):
    same_class_imgs, different_class_imgs = coupler(positive_cores, negative_cores, template_cores)
    x, y = xy_train_creator(same_class_imgs, different_class_imgs)
    #creating siamese network input pairs
    l_input = []
    r_input = []
    labels = []
    length = len(x)
    for i in range(length):
        
        r_index = random.randint(0, len(x)-1) 
        item = x.pop(r_index)
        label = y.pop(r_index)
        l_input.append(item[0])
        r_input.append(item[1])
        labels.append(label)
    
    l_input = np.squeeze(np.array(l_input))
    r_input = np.squeeze(np.array(r_input))
    labels = np.squeeze(np.array(labels))
    
    return l_input, r_input, labels

def xy_train_creator(same_class_imgs, different_class_imgs, balanced = True, verbose = 1):
    #takes the two sets of image couples and merges them
    #note that the training set must be balanced
    #so, i need an equal number of same-class couples and different class couples
    #will also create the Y_train vector
    #the Y_train vector must contain 1 where the sample is made up of same class pictures
    #and 0 where it is made up of different classes
    x_train = []
    y_train = []
    positives = 0
    negatives = 0
    
    #check which of the two sets is the shortest
    l1 = len(same_class_imgs)
    l_2 = len(different_class_imgs)
    if balanced == True: #if I want to have exactly 50 percent same class and 50 percent different class
        
        #get the shortest one 
        if l1 <= l_2:
            #NOTE labeling (1,0) = same class e (0,1) = different class
            #save all elements of data_positive and the same number of elements from data_negative
            for i in range(l1):
                x_train.append(same_class_imgs[i])
                y_train.append((1,0)) #meaning that they belong to the same class
                positives +=1
            
            for i in range(l1):
                x_train.append(different_class_imgs[i])
                y_train.append((0,1)) #meaning that they do not belong to the same class
                negatives +=1
        else:
            for i in range(l_2):
                x_train.append(same_class_imgs[i])
                positives +=1
                y_train.append((1,0)) #meaning that they belong to the same class
            for i in range(l_2):
                x_train.append(different_class_imgs[i])
                y_train.append((0,1)) #meaning that they do not belong to the same class
                negatives +=1
    else:
        #save all elements of data_positive and the same number of elements from data_negative
        for i in range(l1):
            x_train.append(same_class_imgs[i])
            y_train.append((1,0)) #meaning that they belong to the same class
            positives +=1
        for i in range(l_2):
            x_train.append(different_class_imgs[i])
            y_train.append((0,1)) #meaning that they do not belong to the same class
            negatives +=1
    if verbose >0:
        print("Positive tuples: {} | Negative tuples: {} | Total: {}".format(positives,negatives, positives+negatives))
        
    return x_train, y_train

def siamese_nw_multiscala(input_shape):
    
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    #TODO CHECK EFFECT OF DIFFERENT ACTIVATIONS
    activation_inner = "relu" 
    activation_fc = "relu" 
    
    #TODO check effect of regularization
    dropout = False
    norm_1 = False
    norm_2 = False 
    norm_3 = False
    norm_4 = False
    norm_5 = False
    
    ks_1_dim = 9
    ks_2_dim = 5
    ks_3_dim = 3
    
    ks_1 =(ks_1_dim,ks_1_dim)
    ks_2 = (ks_2_dim, ks_2_dim)
    ks_3 = (ks_3_dim, ks_3_dim)
    
    f = 16
    
    #inspired by VGG16 from https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    
    ###Scala 1
    model_scale1 = Sequential()
    
    model_scale1.add(Conv2D(input_shape=input_shape,filters=f,kernel_size=ks_1,padding="same", activation=activation_inner))
    model_scale1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model_scale1.add(Conv2D(filters=f*2, kernel_size=ks_1, padding="same", activation=activation_inner))
    model_scale1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model_scale1.add(Flatten())
    
    
    #scala 2
    model_scale2 = Sequential()
    
    model_scale2.add(Conv2D(input_shape=input_shape,filters=f,kernel_size=ks_2,padding="same", activation=activation_inner))
    model_scale2.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model_scale2.add(Conv2D(filters=f*2, kernel_size=ks_2, padding="same", activation=activation_inner))
    model_scale2.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model_scale2.add(Flatten())
    
    #Scala 3
    model_scale3 = Sequential()
    
    model_scale3.add(Conv2D(input_shape=input_shape,filters=f,kernel_size=ks_3,padding="same", activation=activation_inner))
    model_scale3.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model_scale3.add(Conv2D(filters=f*2, kernel_size=ks_3, padding="same", activation=activation_inner))
    model_scale3.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model_scale3.add(Flatten())
        
    #model.add(Dense(units = 1000, activation=activation_fc))
    
    encoded_l_s1 = model_scale1(left_input)
    encoded_r_s1 = model_scale1(right_input)
    
    encoded_l_s2 = model_scale2(left_input)
    encoded_r_s2 = model_scale2(right_input)
    
    encoded_l_s3 = model_scale3(left_input)
    encoded_r_s3 = model_scale3(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    
    L1_distance_s1 = L1_layer([encoded_l_s1, encoded_r_s1])
    L1_distance_s2 = L1_layer([encoded_l_s2, encoded_r_s2])
    L1_distance_s3 = L1_layer([encoded_l_s3, encoded_r_s3])
    
    concatenated_distances = Concatenate()([L1_distance_s1, L1_distance_s2, L1_distance_s3])
    
    # Add a dense layer with softmax units to generate output probabilities
    prediction = Dense(2,activation='softmax')(concatenated_distances)
    
    siamese_network = Model(inputs=[left_input,right_input],outputs=prediction)
    
    return siamese_network

def siamese_nw(input_shape):
    
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    #TODO CHECK EFFECT OF DIFFERENT ACTIVATIONS
    activation_inner = "relu" 
    activation_fc = "relu" 
    
    #TODO check effect of regularization
    dropout = False
    norm_1 = False
    norm_2 = False 
    norm_3 = False
    norm_4 = False
    norm_5 = False
    
    ks_1_dim = 9
    ks_2_dim = 5
    ks_3_dim = 3
    
    ks_1 =(ks_1_dim,ks_1_dim)
    ks_2 = (ks_2_dim, ks_2_dim)
    ks_3 = (ks_3_dim, ks_3_dim)
    
    f = 16
    
    #inspired by VGG16 from https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    
    model = Sequential()
    
    model.add(Conv2D(input_shape=input_shape,filters=f,kernel_size=ks_1,padding="same", activation=activation_inner))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=f*2, kernel_size=ks_2, padding="same", activation=activation_inner))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=f*4, kernel_size=ks_3, padding="same", activation=activation_inner))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with softmax units to generate output probabilities
    prediction = Dense(2,activation='softmax')(L1_distance)
    
    siamese_network = Model(inputs=[left_input,right_input],outputs=prediction)
    
    return siamese_network

def str_to_bool(string, argname):
    if isinstance(string, bool): #check if input is already a boolean
        return string
    else:
        lowercase_string = string.lower() #convert to lowercase to check for less words
        if lowercase_string in ["true", "yes", "y", "t", "whynot", "1", "ok"]:
            return True
        elif lowercase_string in ["false", "no", "n", "f", "nope", "0" "not"]:
            return False
        else:
            raise ap.ArgumentTypeError("Boolean value expected for {}".format(argname))
            
#load setup
#loading images and creating training image couples
parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required = True, help = "Path to config json.")

args = vars(parser.parse_args())
json_path = args["path_to_json"]

with open(json_path) as f:
    run_params = json.load(f)

force_hosvd = str_to_bool(run_params["force_hosvd"], "Force HOSVD calculation")

template_img = run_params["template"]
mtag = run_params["model_tag"]

train_positive_dir = run_params["train_positive"]
train_negative_dir = run_params["train_negative"]

val_positive_dir = run_params["val_negative"]
val_negative_dir = run_params["val_negative"]

test_positive_dir = run_params["test_negative"]
test_negative_dir = run_params["test_negative"]

hosvd_savefolder = run_params["hosvd_savefolder"]

if not os.path.isdir(hosvd_savefolder):
    force_hosvd = True #savefolder did not exist, hosvd was not calculated/did not complete
    os.mkdir(hosvd_savefolder)
    
template_hosvd_path = os.path.join(hosvd_savefolder, os.path.basename(template_img))
train_positive_hosvd_dir = os.path.join(hosvd_savefolder, os.path.basename(train_positive_dir))
train_negative_hosvd_dir = os.path.join(hosvd_savefolder, os.path.basename(train_negative_dir))

test_positive_hosvd_dir = os.path.join(hosvd_savefolder, os.path.basename(test_positive_dir))
test_negative_hosvd_dir = os.path.join(hosvd_savefolder, os.path.basename(test_negative_dir))

dirs_to_test = [template_hosvd_path, train_positive_hosvd_dir, train_negative_hosvd_dir,
                test_positive_hosvd_dir, test_negative_hosvd_dir]

if (val_positive_dir != "None") or (val_negative_dir != "None"):
    val_positive_hosvd_dir = os.path.join(hosvd_savefolder, os.path.basename(val_positive_dir))
    val_negative_hosvd_dir = os.path.join(hosvd_savefolder, os.path.basename(val_negative_dir))
    
    dirs_to_test = [template_hosvd_path, train_positive_hosvd_dir, train_negative_hosvd_dir,
                    val_positive_hosvd_dir, val_negative_hosvd_dir,
                    test_positive_hosvd_dir, test_negative_hosvd_dir]
    
saved_model_name = os.path.join(run_params["model_savedir"],  str(mtag) +"_trained.h5")

hosvd_checks = []
for item in dirs_to_test:
    #check if directory exists
    if os.path.isdir(item):
        #check if directory is not empty (per ora si assume che se viene riempita l'hosvd è andata a buon fine, #TODO raffinabile?)
        if len(os.listdir(item)) != 0:
            #if the directory is not empty but force training is selected, delete it and empty it
            if force_hosvd:
                shutil.rmtree(item)
                os.mkdir(item)
                hosvd_checks.append(False)
            else:
                hosvd_checks.append(True)
        else:
            hosvd_checks.append(False)
    else:
        #directory does not exist, hosvd will be recalculated, #TODO è possibile rifare l'hosvd solo per le cartelle che mancano? BOH
        hosvd_checks.append(False)
        os.mkdir(item)

#usage parameters
img_dim = run_params["resize_img_dim"]
compression = run_params["compression"]
hosvd_channels = run_params["hosvd_channels"]

d1 = int(img_dim/compression)
tucker_rank = [d1,d1,hosvd_channels]
input_dim = tucker_rank

#print(np.shape(imgs[0]))
#do the hosvd with tucker decomposition
#check if hosvd must be calculated/recalculated or not:
if force_hosvd or not all(hosvd_checks):
    #Loading data
    print("Loading and resizing:")
    print("Template images")
    template_imgs, filenames_template = load_images_from_folder(template_img, img_dim)
    print("Training images")
    training_positive_imgs, filenames_training_positive = load_images_from_folder(train_positive_dir,img_dim)
    training_negative_imgs, filenames_training_negative = load_images_from_folder(train_negative_dir,img_dim)

    if (val_positive_dir != "None") or (val_negative_dir != "None"):
        print("Validation images")
        val_positive_imgs, filenames_val_positive = load_images_from_folder(val_positive_dir,img_dim)
        val_negative_imgs, filenames_val_negative = load_images_from_folder(val_negative_dir,img_dim)
        
    print("Test images")
    test_positive_imgs, filenames_test_positive = load_images_from_folder(test_positive_dir,img_dim)
    test_negative_imgs, filenames_test_negative = load_images_from_folder(test_negative_dir,img_dim)
    
    print("Peforming HOSVD with Tucker decomposition for:")
    print("Core tensors will be rescaled with L2 norm")
    
    print("Template images")
    #decompose
    template_cores = tucker_decomposed_imgs(template_imgs, tucker_rank)
    #rescale
    template_cores = rescale_cores(template_cores)
    #saving template cores
    save_matrices(template_cores, filenames_template, template_hosvd_path)
    
    print("Training images")
    #decompose
    training_positive_cores = tucker_decomposed_imgs(training_positive_imgs, tucker_rank)
    training_negative_cores = tucker_decomposed_imgs(training_negative_imgs, tucker_rank)
    #rescale
    training_positive_cores = rescale_cores(training_positive_cores)
    training_negative_cores = rescale_cores(training_negative_cores)
    #saving training cores
    save_matrices(training_positive_cores, filenames_training_positive, train_positive_hosvd_dir)
    save_matrices(training_negative_cores, filenames_training_negative, train_negative_hosvd_dir)
    
    if (val_positive_dir != "None") or (val_negative_dir != "None"):
        print("Validation images")
        #decompose
        val_positive_cores = tucker_decomposed_imgs(val_positive_imgs, tucker_rank)
        val_negative_cores = tucker_decomposed_imgs(val_negative_imgs, tucker_rank)
        #rescale
        val_positive_cores = rescale_cores(val_positive_cores)
        val_negative_cores = rescale_cores(val_negative_cores)
        #save validation cores
        save_matrices(val_positive_cores, filenames_test_positive, val_positive_hosvd_dir)
        save_matrices(val_negative_cores, filenames_test_negative, val_negative_hosvd_dir)
        
    print("Test images")
    test_positive_cores = tucker_decomposed_imgs(test_positive_imgs, tucker_rank)
    test_negative_cores = tucker_decomposed_imgs(test_negative_imgs, tucker_rank)
    #rescale
    test_positive_cores = rescale_cores(test_positive_cores)
    test_negative_cores = rescale_cores(test_negative_cores)
    #saving test cores
    save_matrices(test_positive_cores, filenames_test_positive, test_positive_hosvd_dir)
    save_matrices(test_negative_cores, filenames_test_negative, test_negative_hosvd_dir)
    
else:
    print("Reloading old HOSVD results...")
    #reload the results of previous HOSVD calculations
    template_cores = load_matrices(template_hosvd_path)
    training_positive_cores = load_matrices(train_positive_hosvd_dir)
    training_negative_cores = load_matrices(train_negative_hosvd_dir)
    if (val_positive_dir != "None") or (val_negative_dir != "None"):
        val_positive_cores = load_matrices(val_positive_hosvd_dir)
        val_negative_cores = load_matrices(val_negative_hosvd_dir)
    test_positive_cores = load_matrices(test_positive_hosvd_dir)
    test_negative_cores = load_matrices(test_negative_hosvd_dir)

# #create training tuples
print("Creating siamese network input tuples for:")
print("Training set")
l_input_train, r_input_train, labels_train = siamese_input_creator(training_positive_cores, training_negative_cores, template_cores)
if (val_positive_dir != "None") or (val_negative_dir != "None"):
    print("Validation images")
    l_input_val, r_input_val, labels_val = siamese_input_creator(val_positive_cores, val_negative_cores, template_cores)
print("Test set")
l_input_test, r_input_test, labels_test = siamese_input_creator(test_positive_cores, test_negative_cores, template_cores)


#run parameters
generator_chunk = 64
enable_early_stopping = True 
show_misclassified_images = False
stat_output = True
reload_top = False
#training parameters
l_r = 5e-3
opt = Nadam(learning_rate = l_r)
batch_size = 128
epochs = 32
loss = "hinge"
metric = ["accuracy"]

print("Done, starting learning.\n")
#defining a simple early stopping callback and a checkpoint one
es = EarlyStopping( monitor = "val_loss", mode = "min", verbose = 1, patience = 2)
mc = ModelCheckpoint(saved_model_name, monitor = "val_loss", mode = "min", verbose = 1, save_best_only = True) #will save the best performing model overall

if enable_early_stopping == True:
    cbs = [es, mc]

else:
    cbs = []
    
#defining and compiling model
model = siamese_nw(input_dim)

print("Model summary:")
model.summary()
print()
model.compile(loss= loss, optimizer=opt, metrics= metric)

#training model
if (val_positive_dir != "None") or (val_negative_dir != "None"):
    # history = model.fit([l_input, r_input], label_train, validation_data = ([l_input_val, r_input_val], label_val), epochs = epochs,
    #                     batch_size = batch_size, callbacks=cbs)
    history = model.fit(siamese_input_gen(l_input_train, r_input_train, labels_train, n= generator_chunk), validation_data = siamese_input_gen(l_input_val, r_input_val, labels_val, n= generator_chunk), epochs = epochs,
                        batch_size = batch_size, callbacks=cbs)
else:
    # history = model.fit([l_input, r_input], label_train, epochs = epochs,
    #                     batch_size = batch_size, callbacks=cbs)
    history = model.fit(siamese_input_gen(l_input_train, r_input_train, labels_train, n= generator_chunk), epochs = epochs,
                        batch_size = batch_size, callbacks=cbs)

#save model
model.save(saved_model_name)





























