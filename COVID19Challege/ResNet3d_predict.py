import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from ResNet2d.model_resNet2d import ResNet2dModule
import cv2
import pandas as pd
import numpy as np


def predict():
    test_COVID = pd.read_csv('dataprocess\\data\\test_COVID.csv')
    test_NoCOVID = pd.read_csv('dataprocess\\data\\test_NoCOVID.csv')
    test_COVIDdata = test_COVID.iloc[:, :].values
    test_NoCOVID = test_NoCOVID.iloc[:, :].values
    testData = np.concatenate((test_COVIDdata, test_NoCOVID), axis=0)
    # For Image
    images = testData[:, 1]
    # For Labels
    labels = testData[:, 0]
    ResNet2d = ResNet2dModule(256, 256, channels=1, n_class=2, costname="cross_entropy", inference=True,
                              model_path="log\COVID19/resnet\model/resnet.pd-20000")

    predictvalues = []
    predict_probs = []
    for num in range(np.shape(images)[0]):
        batchimage = np.reshape(cv2.imread("dataprocess\\" + images[num], 0), (1, 256, 256, 1))
        inpuimage = batchimage.astype(np.float)
        # Normalize from [0:255] => [0.0:1.0]
        inpuimage = np.multiply(inpuimage, 1.0 / 255.0)
        predictvalue, predict_prob = ResNet2d.prediction(inpuimage)
        predictvalues.append(predictvalue)
        predict_probs.append(predict_prob)

    name = 'classify_metrics.csv'
    out = open(name, 'w')
    out.writelines("y_predict" + "," + "y_score" + "," + "y_true" + "\n")
    labels = labels.tolist()
    for index in range(np.shape(images)[0]):
        out.writelines(
            str(predictvalues[index][0]) + "," + str(predict_probs[index][0]) + "," + str(labels[index]) + "\n")


if __name__ == "__main__":
    predict()
