import os
import cv2
import numpy as np

def normalize_training_data(cfg):
    TrainFolder=cfg["data"].get("path")
    ListImages=os.listdir(os.path.join(TrainFolder, "lytro-img/A")) # Create list of images
    # Initialize empty lists to store pixel values for each channel
    channel_red = []
    channel_green = []
    channel_blue = []

    for idx in range(len(ListImages)):
        Img1=cv2.imread(os.path.join(TrainFolder, "lytro-img/A", ListImages[idx]).replace("._", ""))[:,:,0:3]/255.0
        Img2=cv2.imread(os.path.join(TrainFolder, "lytro-img/B", ListImages[idx]).replace("A", "B").replace("._", ""))[:,:,0:3]/255.0

        channel_red.extend(Img1[:, :, 0].ravel())
        channel_green.extend(Img1[:, :, 1].ravel())
        channel_blue.extend(Img1[:, :, 2].ravel())
        channel_red.extend(Img2[:, :, 0].ravel())
        channel_green.extend(Img2[:, :, 1].ravel())
        channel_blue.extend(Img2[:, :, 2].ravel())

    std = (np.std(channel_red), np.std(channel_green), np.std(channel_blue))
    mean = (np.mean(channel_red), np.mean(channel_green), np.mean(channel_blue))
    return mean, std


def normalize_pair(Img1, Img2):
    # Initialize empty lists to store pixel values for each channel
    channel_red = []
    channel_green = []
    channel_blue = []

    Img1 = Img1/255.0
    Img1 = Img1/255.0
    
    channel_red.extend(Img1[:, :, 0].ravel())
    channel_green.extend(Img1[:, :, 1].ravel())
    channel_blue.extend(Img1[:, :, 2].ravel())
    channel_red.extend(Img2[:, :, 0].ravel())
    channel_green.extend(Img2[:, :, 1].ravel())
    channel_blue.extend(Img2[:, :, 2].ravel())

    std = (np.std(channel_red), np.std(channel_green), np.std(channel_blue))
    mean = (np.mean(channel_red), np.mean(channel_green), np.mean(channel_blue))
    return mean, std