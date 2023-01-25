# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:41:49 2023

@author: ecero
"""

### training script for siamese network training
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse as ap
import json
import cv2

#setup gpu 
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Nadam

from keras.layers import BatchNormalization, Concatenate
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPool2D, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K

from sklearn.metrics import confusion_matrix

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def coupler(positive_image_dir, negative_image_dir, template_img, new_dim):
    data_positive = []
    #loads the template(s)
    for file in os.listdir(template_img):
        img = cv2.imread(os.path.join(template_img, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #now resize
        try:
            img = cv2.resize(img, (new_dim, new_dim))
            data_positive.append(img)
        except:
            continue
    num_template = len(data_positive)
    for file in os.listdir(positive_image_dir):
        
        img = cv2.imread(os.path.join(positive_image_dir, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            img = cv2.resize(img, (new_dim, new_dim))
            data_positive.append(img)
        except:
            continue
        
    data_negative = []
    
    for file in os.listdir(negative_image_dir):
        
        img = cv2.imread(os.path.join(negative_image_dir, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            img = cv2.resize(img, (new_dim, new_dim))
            data_negative.append(img)
        except:
            continue
    same_class_imgs = []
    different_class_imgs = []
    max_coupleable = num_template  #TODO set this as parameters for function
   
    for i in range(len(data_positive) - 1): #
        if i <= max_coupleable:
            sample = data_positive[i]
            remaining = data_positive[i:]
            
            for item in remaining:
                couple = []
                couple.append(sample)
                couple.append(item)
                same_class_imgs.append(couple)
            for item in data_negative:
                couple = []
                couple.append(sample)
                couple.append(item)
                different_class_imgs.append(couple)
    print("Number of same class image pairs = {}, Number of different class image pairs = {}, total sample pairs: {}\n".format(len(same_class_imgs), len(different_class_imgs), len(same_class_imgs) + len(different_class_imgs) ))
    return same_class_imgs, different_class_imgs

def xy_train_creator(same_class_imgs, different_class_imgs, balanced = False):
    #takes the two sets of image couples and merges them
    #note that the training set must be balanced
    #so, i need an equal number of same-class couples and different class couples
    #will also create the Y_train vector
    #the Y_train vector must contain 1 where the sample is made up of same class pictures
    #and 0 where it is made up of different classes
    x_train = []
    y_train = []
    
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
            
            for i in range(l1):
                x_train.append(different_class_imgs[i])
                y_train.append((0,1)) #meaning that they do not belong to the same class
        else:
            for i in range(l_2):
                x_train.append(same_class_imgs[i])
                y_train.append((1,0)) #meaning that they belong to the same class
            for i in range(l_2):
                x_train.append(different_class_imgs[i])
                y_train.append((0,1)) #meaning that they do not belong to the same class
    else:
        #save all elements of data_positive and the same number of elements from data_negative
        for i in range(l1):
            x_train.append(same_class_imgs[i])
            y_train.append((1,0)) #meaning that they belong to the same class
        for i in range(l_2):
            x_train.append(different_class_imgs[i])
            y_train.append((0,1)) #meaning that they do not belong to the same class
    return x_train, y_train



def some_stats(y_pred, rounded, label_test):
    
    counter_correct = 0
    counter_misclassified = 0
    counter_misclassified_positive = 0
    counter_misclassified_negative = 0
    
    #same class pictures are labeled as (1,0)
    mis_positive_indxs = []
    mis_negative_indxs = []
    
    for i in range(len(y_pred)):
        confidences = y_pred[i]
        predicted = rounded[i]
        real = label_test[i]
        if predicted[0] == real[0] and predicted[1] == real[1]:
            counter_correct += 1
            #status = "Correct classification. Output: Same class prob: {}, diff class prob: {}".format(round(confidences[0]*100, 2), round(confidences[1]*100, 2))
        else:
            counter_misclassified += 1
            if predicted[0] == 0 and real[0] == 1: #this means that a positive positive pair has been classified as a positive- adult, worst error case scenario
                counter_misclassified_positive += 1
                mis_positive_indxs.append(i)
                #status = "Misclassified positive, troublesome"
            else:
                #this means that i misclassidied a positive-not positive pair as a positive-positive one, small problem
                counter_misclassified_negative += 1
                mis_negative_indxs.append(i)
                #status = "Misclassified a non positive, not a problem"
        
        #print("Predicted label (rounded): {}, True label: {}, Status: {}".format(predicted, real, status))
    
    plot_test = []
    plot_pred = []
    
    #will take only first 25 results, otherwise the plot gets messy
    for i in label_test:
        plot_test.append(i[0])
    
    for i in rounded:
        plot_pred.append(i[0])
        
    cm = confusion_matrix(plot_test,plot_pred)
    print("Correctly classified examples: {}, Misclassified examples: {}, Misclassified positive: {}, Misclassified non positive: {}".format(counter_correct, counter_misclassified, counter_misclassified_positive, counter_misclassified_negative ))
    print("Confusion Matrix:\n")
    print(cm)
    
    return mis_positive_indxs, mis_negative_indxs

#input_shape = (dim,dim,3)

def siamese_nw(input_shape):
    
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    #TODO CHECK EFFECT OF DIFFERENT ACTIVATIONS
    activation_inner = "tanh" 
    activation_fc = "relu" 
    
    #TODO check effect of regularization
    dropout = False
    norm_1 = False
    norm_2 = False 
    norm_3 = False
    norm_4 = False
    norm_5 = False
    
    
    #inspired by VGG16 from https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    model = Sequential()
    
    model.add(Conv2D(input_shape=(112,112,3),filters=64,kernel_size=(3,3),padding="same", activation=activation_inner))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation=activation_inner))
    
    if norm_1 == True:
        model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    if dropout == True:
        
        model.add(Dropout(0.1))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    
    if norm_2 == True:
        model.add(BatchNormalization())
        
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # if dropout == True:
        
    #     model.add(Dropout(0.1))
    
    # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    
    # if norm_3 == True:
        
    #     model.add(BatchNormalization())
        
    # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # if dropout == True:
        
    #     model.add(Dropout(0.1))
    
    # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_inner))
    
    # if norm_4 == True:
        
    #     model.add(BatchNormalization())
        
    # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # if dropout == True:
    #     model.add(Dropout(0.05))
    
    # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=activation_inner))
    # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=activation_inner))
    # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=activation_inner))
    
    # if norm_5 == True:
        
    #     model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # if dropout == True:
        
    #     model.add(Dropout(0.05))
    
    model.add(Flatten())
        
    model.add(Dense(units = 1000, activation=activation_fc))
    
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with softmax units to generate output probabilities
    prediction = Dense(2,activation='softmax')(L1_distance)
    
    siamese_network = Model(inputs=[left_input,right_input],outputs=prediction)
    
    return siamese_network

#setting up
#loading images and creating training image couples
parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required = True, help = "Path to config json.")

args = vars(parser.parse_args())
params = args["path_to_json"]

with open(params) as f:
    dataset_dict = json.load(f)

template_img = dataset_dict["template"]
mtag = dataset_dict["model_tag"]

#usage parameters
new_dim = 128
input_dim = (new_dim, new_dim, 3)

#run parameters
enable_early_stopping = True 
show_misclassified_images = False
stat_output = True
reload_top = False
#training parameters
l_r = 5e-4
opt = Nadam(lr = l_r)
batch_size = 128
epochs = 20
loss = "hinge"
metric = ["accuracy"]

saved_model_name = os.path.join(dataset_dict["model_savedir"],  str(mtag) +"_trained.h5")

train_positive_dir = dataset_dict["train_positive"]
train_negative_dir = dataset_dict["train_negative"]

val_positive_dir = dataset_dict["val_negative"]
val_negative_dir = dataset_dict["val_negative"]

test_positive_dir = dataset_dict["test_negative"]
test_negative_dir = dataset_dict["test_negative"]

print("Training set: ")
same_class_imgs, different_class_imgs = coupler(train_positive_dir, train_negative_dir, template_img, new_dim)
x, y = xy_train_creator(same_class_imgs, different_class_imgs)

#creating siamese network input pairs
l_input = []
r_input = []
label_train = []
length = len(x)
for i in range(length):
    
    r_index = random.randint(0, len(x)-1) 
    item = x.pop(r_index)
    label = y.pop(r_index)
    l_input.append(item[0])
    r_input.append(item[1])
    label_train.append(label)

l_input = np.squeeze(np.array(l_input))
r_input = np.squeeze(np.array(r_input))
label_train = np.squeeze(np.array(label_train))

if (val_positive_dir != "None") or (val_negative_dir != "None"):
    print("Validation set: ")
    val_same_class_imgs, val_different_class_imgs = coupler(val_positive_dir, val_negative_dir, template_img, new_dim)
    x_v, y_v = xy_train_creator(val_same_class_imgs, val_different_class_imgs)
    
    l_input_val = []
    r_input_val = []
    label_val = []
    
    length_val = len(x_v)
    for i in range(length_val):
        
        r_index = random.randint(0, len(x_v)-1)
        
        item = x_v.pop(r_index)
        label = y_v.pop(r_index)
        
        l_input_val.append(item[0])
        r_input_val.append(item[1])
        label_val.append(label)
        
    l_input_val = np.squeeze(np.array(l_input_val))
    r_input_val = np.squeeze(np.array(r_input_val))
    label_val = np.squeeze(np.array(label_val))
    
    
print("Test set: ")
test_same_class_imgs, test_different_class_imgs = coupler(test_positive_dir, test_negative_dir, template_img, new_dim)
x_t, y_t =  xy_train_creator(test_same_class_imgs, test_different_class_imgs)

l_input_test = []
r_input_test = []
label_test  =[]
length_test = len(x_t)
for i in range(length_test):
    
    r_index = random.randint(0, len(x_t)-1)
    
    item = x_t.pop(r_index)
    label = y_t.pop(r_index)
    
    l_input_test.append(item[0])
    r_input_test.append(item[1])
    label_test.append(label)

l_input_test = np.squeeze(np.array(l_input_test))
r_input_test = np.squeeze(np.array(r_input_test))
label_test = np.squeeze(np.array(label_test))

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
    history = model.fit([l_input, r_input], label_train, validation_data = ([l_input_val, r_input_val], label_val), epochs = epochs,
                        batch_size = batch_size, callbacks=cbs)
else:
    history = model.fit([l_input, r_input], label_train, epochs = epochs,
                        batch_size = batch_size, callbacks=cbs)

#save model
model.save(saved_model_name)

#shows how the loss and accuracy evolved during training on train and validation set
plt.plot(history.history['accuracy'])
if (val_positive_dir != "None") or (val_negative_dir != "None"):
    plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
if (val_positive_dir != "None") or (val_negative_dir != "None"):
    plt.legend(['Train', 'Val'], loc = 'lower right')
else:
    plt.legend(['Train'], loc = 'lower right')
plt.show()


plt.plot(history.history['loss'])
if (val_positive_dir != "None") or (val_negative_dir != "None"):
    plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
if (val_positive_dir != "None") or (val_negative_dir != "None"):
    plt.legend(['Train', 'Val'], loc = 'lower right')
else:
    plt.legend(['Train'], loc = 'lower right')
plt.show()

if enable_early_stopping == True:
    
    #i found and saved a best model, will reload it
    
    if reload_top == True:
        print("Loading best known model")
        reloaded = load_model("rgb_only_faces_lateral_templates_best_model_mlp.h5")
        print("Done")
    else:
        print("Loading best model now")
        reloaded = load_model(saved_model_name)
        print("Done")

print("Making predictions on test set")
y_pred = reloaded.predict([l_input_test, r_input_test], verbose = 1)
print("Done")

rounded = [] #will be used to make a decision, as output of the network

for item in y_pred:
    
    x_label = item[0]
    rounded_x = round(x_label)
    y_label = item [1]
    rounded_y = round(y_label)
    
    rounded_tuple = [rounded_x, rounded_y]
    rounded.append(rounded_tuple)

#get some statistics printed on screen
if stat_output == True:
    
    mis_positive_indxs, mis_negative_indxs = some_stats(y_pred, rounded, label_test)

print("Evaluating model")
results = reloaded.evaluate([l_input_test, r_input_test], label_test, verbose = 1)
print('Test loss: {}, Test accuracy: {}'.format(results[0], round(results[1]*100 , 2)))

