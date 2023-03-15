# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:11:02 2023

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
import pandas as pd
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def get_mobilenet_classifier(input_shape, trainable_base = False, lr = 1e-3):
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
    
    outputs = keras.layers.Dense(1)(base_model_encoding)

    #defining model
    model = keras.Model(inputs, outputs)
    #compiling the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = lr),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])
    return model

def get_mobilenet_encoder(input_shape, trainable_base = False, lr = 1e-3):
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
    
    outputs = keras.layers.Dense(1)(base_model_encoding)
    
    #defining model
    model = keras.Model(inputs, base_model_encoding)
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

if not os.path.isdir("mnet_encodings"):
    os.mkdir("mnet_encodings")
    
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

reload_base = False
reload_base_ft = False

# if reload_base:
#     model = get_mobilenet_classifier(model_input_shape, trainable_base = False, lr = run_params["top_model_optimizer_learning_rate"])
#     model.load_weights(os.path.join("checkpoints", "MobileNetFrozen_encClass_preFineTuning_imgs_weights"))
    
# else:
#     #get frozen model
#     model = get_mobilenet_classifier(model_input_shape, trainable_base = False, lr = run_params["top_model_optimizer_learning_rate"])
#     print("Model summary")
#     print(model.summary())
#     print("Training the classifier on top of MobileNetV3")
#     # Train the model in batches 
#     idx = 1
#     for images, labels in train_dataset:
#         print("Training batch {} out of {}".format(idx, len(train_dataset)))
#         model.fit(x = images, y = labels, epochs=run_params["top_model_epochs"],shuffle = True)
#         if idx > 4:
#             break
#         idx +=1
        
#     model.save_weights(os.path.join("checkpoints", "MobileNetFrozen_encClass_preFineTuning_imgs_weights"))
    
# del model
# sess = tf.compat.v1.keras.backend.get_session()
# tf.compat.v1.keras.backend.clear_session()
# sess.close()
# sess = tf.compat.v1.keras.backend.get_session()


# gc.collect()

# config = tf.compat.v1.ConfigProto()
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# #fine tuning della mobile net
# if not reload_base_ft:
#     #get frozen model
#     model = get_mobilenet_classifier(model_input_shape, trainable_base = True, lr = run_params["full_model_optimizer_learning_rate"])
#     model.load_weights(os.path.join("checkpoints", "MobileNetFrozen_encClass_preFineTuning_imgs_weights"))
    
#     print("Model summary")
#     print(model.summary())
#     print("Training the classifier on top of MobileNetV3")
#     # Train the model in batches 
#     idx = 1
#     for images, labels in train_dataset:
#         print("Training batch {} out of {}".format(idx, len(train_dataset)))
#         model.fit(x = images, y = labels, epochs=run_params["top_model_epochs"],shuffle = True)
#         idx +=1
        
#     model.save_weights(os.path.join("checkpoints", "MobileNetFrozen_encClass_postFineTuning_imgs_weights"))

# del model
# sess = tf.compat.v1.keras.backend.get_session()
# tf.compat.v1.keras.backend.clear_session()
# sess.close()
# sess = tf.compat.v1.keras.backend.get_session()


# gc.collect()

# config = tf.compat.v1.ConfigProto()
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


encoder = get_mobilenet_encoder(model_input_shape, trainable_base = True)
encoder.load_weights(os.path.join("checkpoints", "MobileNetFrozen_encClass_postFineTuning_imgs_weights"))

classifier = get_mobilenet_classifier(model_input_shape, trainable_base = True)
classifier.load_weights(os.path.join("checkpoints", "MobileNetFrozen_encClass_postFineTuning_imgs_weights"))

idx = 1
cols = [f'F{i+1}' for i in range(960)]
cols.append("Label")
train_filepath = os.path.join("mnet_encodings", "StyleGAN2_train_encoded.csv")
if os.path.exists(train_filepath):
    train_encoded_vectors = pd.read_csv(train_filepath)
else:
    train_encoded_vectors = pd.DataFrame(columns=cols) #960 è la dimensione del vettore di output di mobilenetv3
    for train_images, train_labels in train_dataset:
        print("Train batch {} out of {}".format(idx, len(train_dataset)))
        # add arr2 as a column to arr1
        encoded = encoder.predict(train_images)
        encoded_with_labels = np.hstack((encoded, train_labels.numpy()[:, np.newaxis]))
        train_encoded_vectors = train_encoded_vectors.append(pd.DataFrame(encoded_with_labels,
                                                                          columns=train_encoded_vectors.columns),
                                                                         ignore_index=True)
        idx +=1
    
    
    train_encoded_vectors.to_csv(train_filepath, sep = ",", index = False)

idx = 1
test_filepath = os.path.join("mnet_encodings", "StyleGAN3_encoded.csv")
if os.path.exists(test_filepath):
    test_encoded_vectors = pd.read_csv(test_filepath)
else:
    
    test_encoded_vectors = pd.DataFrame(columns=cols)
    for test_images, test_labels in test_dataset:
        print("Test batch {} out of {}".format(idx, len(test_dataset)))
        encoded = encoder.predict(test_images)
        encoded_with_labels = np.hstack((encoded, test_labels.numpy()[:, np.newaxis]))
        test_encoded_vectors = test_encoded_vectors.append(pd.DataFrame(encoded_with_labels,
                                                                        columns=test_encoded_vectors.columns),
                                                                        ignore_index=True)
        idx +=1
    
    
    test_encoded_vectors.to_csv(test_filepath, sep = ",", index = False)
    

idx = 1
chunk_val_accuracies = []
for val_images, val_labels in val_dataset:
    print("Val batch {} out of {}".format(idx, len(val_dataset)))
    chunk_val_accuracies.append(classifier.evaluate(val_images,val_labels))
    idx +=1

for i in range(len(chunk_val_accuracies)):
    print("Val Chunk {} accuracy: {}".format(i+1,chunk_val_accuracies[i]))
    
idx = 1
chunk_test_accuracies = []
for test_images, test_labels in test_dataset:
    print("Test batch {} out of {}".format(idx, len(test_dataset)))
    chunk_test_accuracies.append(classifier.evaluate(test_images,test_labels))
    idx +=1

for i in range(len(chunk_test_accuracies)):
    print("Test Chunk {} accuracy: {}".format(i+1,chunk_test_accuracies[i]))
    
# train_encoded_vectors_only_real = train_encoded_vectors.loc[train_encoded_vectors["Label"] == 0].copy()
test_encoded_vectors_only_sint = test_encoded_vectors.loc[test_encoded_vectors["Label"] == 1].copy()

#TODO umap e plot proiezione

# seleziona solo le colonne con i dati (ignorando l'ultima colonna con le label)
test_encoded_vectors_only_sint.loc[test_encoded_vectors_only_sint['Label'] ==1, 'Label'] = 2

# all_vectors = pd.concat([train_encoded_vectors_only_real, test_encoded_vectors_only_sint], ignore_index=True)
all_vectors = pd.concat([train_encoded_vectors, test_encoded_vectors_only_sint], ignore_index=True)

enc_vectors = all_vectors.iloc[:, :-1]


# usa UMAP per ridurre la dimensionalità a 2 componenti
ncomp = 5
reducer = umap.UMAP(n_components=ncomp)
#reducer = PCA(n_components=ncomp)
embedding = reducer.fit_transform(enc_vectors)

# crea un dataframe con i dati proiettati
df = pd.DataFrame(data=embedding, columns=['component_{}'.format(i+1) for i in range(embedding.shape[1])])
df['label'] = all_vectors.iloc[:, -1]

# crea il pairplot
sns.pairplot(data=df, hue='label')
plt.show()
# usa PCA per ridurre la dimensionalità a 2 componenti
# reducer = PCA(n_components=10)
# embedding = reducer.fit_transform(all_vectors)

# # crea un grafico 2D dei dati proiettati, colorati in base alle label
# labels = all_vectors.iloc[:, -1]
# plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
# # crea i marker separati
# markers = []
# for label in set(labels):
#     marker = plt.Line2D([0,0], [0,0], color=plt.cm.jet(label / float(len(set(labels)))), marker='o', linestyle='')
#     markers.append(marker)

# # aggiungi una legenda
# plt.legend(markers, list(set(labels)), loc='lower left')
# plt.show()


























