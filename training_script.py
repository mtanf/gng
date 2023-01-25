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

