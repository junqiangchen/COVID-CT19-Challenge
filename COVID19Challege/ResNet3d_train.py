import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from ResNet2d.model_resNet2d import ResNet2dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    # csv file should have the type:
    # label,data_npy
    # label,data_npy
    # ....
    #
    train_augCOVID = pd.read_csv('dataprocess\\data\\train_augCOVID.csv')
    train_augNoCOVID = pd.read_csv('dataprocess\\data\\train_augNoCOVID.csv')
    train_COVID = pd.read_csv('dataprocess\\data\\train_COVID.csv')
    train_NoCOVID = pd.read_csv('dataprocess\\data\\train_NoCOVID.csv')
    val_COVID = pd.read_csv('dataprocess\\data\\val_COVID.csv')
    val_NoCOVID = pd.read_csv('dataprocess\\data\\val_NoCOVID.csv')

    augCOVIDdata = train_augCOVID.iloc[:, :].values
    augNoCOVID = train_augNoCOVID.iloc[:, :].values
    COVIDdata = train_COVID.iloc[:, :].values
    NoCOVID = train_NoCOVID.iloc[:, :].values

    val_COVIDdata = val_COVID.iloc[:, :].values
    val_NoCOVID = val_NoCOVID.iloc[:, :].values
    valData = np.concatenate((val_COVIDdata, val_NoCOVID), axis=0)

    trainData = np.concatenate((COVIDdata, NoCOVID, augCOVIDdata, augNoCOVID, val_COVIDdata, val_NoCOVID), axis=0)
    np.random.shuffle(trainData)
    # For Image
    trainimages = trainData[:, 1]
    valimages = valData[:, 1]
    # For Labels
    trainlabels = trainData[:, 0]
    vallabels = valData[:, 0]
    ResNet3d = ResNet2dModule(256, 256, channels=1, n_class=2, costname="cross_entropy")
    ResNet3d.train(trainimages, trainlabels, valimages, vallabels, "resnet.pd", "log\\COVID19\\resnet\\", 0.001, 0.5,
                   10, 32)


if __name__ == "__main__":
    train()
