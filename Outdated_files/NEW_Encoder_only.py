# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:03:19 2023

@author: ecero
"""

#mobileNet encoder
#reloads a model and uses it as an encoder
import os
import numpy as np
import argparse as ap
import json
import tensorflow as tf
from tensorflow import keras
import gc
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

#TODO definire una libreria di utils e caricare da li
#function that decodes and loads images
def load_and_decode_image(filename, label, image_size):
    # Load and decode the image from the file
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    # Resize the image to the desired size
    image = tf.image.resize(image, image_size)    
    # Convert label to a tensor
    label = tf.convert_to_tensor(label)
    return image, label

###begin
parser = ap.ArgumentParser()
parser.add_argument("-json", "--path_to_json", required=True, help="Path to config json.")
