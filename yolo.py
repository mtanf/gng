import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_matrices(folder_path):
    reloaded_matrices = []
    filenames = os.listdir(folder_path)
    for filename in filenames:
        reloaded_matrices.append(np.load(os.path.join(folder_path, filename), allow_pickle=False))

    return reloaded_matrices


# Carica il tensore in formato .npy
tensor = np.load('/repo/tanfoni/GNG_dataset/Compressed_imgs/Real_run_4K/01982_OF_20_0_HOSVD_Core.npy')
tensor2= np.load('/repo/tanfoni/GNG_dataset/Compressed_imgs/Sint_run_4K/2JP6I2XEBC_OF_20_0_HOSVD_Core.npy')

img_float32 = np.float32(tensor)
img2_float32 = np.float32(tensor2)
img = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)
img2 = cv2.cvtColor(img2_float32, cv2.COLOR_RGB2HSV)

# Visualizza l'immagine
cv2.imshow('Image', img)
cv2.imshow('Image2', img2)
cv2.waitKey(0)
# cv2.destroyAllWindows()
