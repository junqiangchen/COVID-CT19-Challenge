from __future__ import print_function, division
import cv2
import os
from dataaugmation.Augmentation.ImageAugmentation2d import DataAug2d


def files_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(files):
            print("sub_files:", files)
            return files


def save_file2csv(file_dir, file_name, label):
    """
    save file path to csv,this is for classification
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :param label:classification label
    :return:
    """
    out = open(file_name, 'w')
    sub_files = files_name_path(file_dir)
    out.writelines("class,filename" + "\n")
    for index in range(len(sub_files)):
        out.writelines(str(label) + "," + file_dir + "/" + sub_files[index] + "\n")


def Data_splitintotrainvaltest():
    testCT_NonCOVID = "E:\Data\COVID19\Data-split\\NonCOVID\\testCT_NonCOVID.txt"
    trainCT_NonCOVID = "E:\Data\COVID19\Data-split\\NonCOVID\\trainCT_NonCOVID.txt"
    valCT_NonCOVID = "E:\Data\COVID19\Data-split\\NonCOVID\\valCT_NonCOVID.txt"
    testCT_COVID = "E:\Data\COVID19\Data-split\COVID\\testCT_COVID.txt"
    trainCT_COVID = "E:\Data\COVID19\Data-split\COVID\\trainCT_COVID.txt"
    valCT_COVID = "E:\Data\COVID19\Data-split\COVID\\valCT_COVID.txt"

    COVID_imagePath = "E:\Data\COVID19\Images-processed\CT_COVID\\"
    NonCOVID_imagePath = "E:\Data\COVID19\Images-processed\CT_NonCOVID\\"

    testCT_NonCOVIDfile = open(testCT_NonCOVID, "r")
    testCT_NonCOVIDlines = testCT_NonCOVIDfile.readlines()
    testCT_NonCOVIDfile_path = "data\\test\\NoCOVID\\"
    for line in testCT_NonCOVIDlines:
        if '\n' in line:
            line = line[0:-1]
        imagepath = NonCOVID_imagePath + line
        image = cv2.imread(imagepath, 0)
        imagenew = cv2.resize(image, dsize=(256, 256))
        cv2.imwrite(testCT_NonCOVIDfile_path + line, imagenew)

    trainCT_NonCOVIDfile = open(trainCT_NonCOVID, "r")
    trainCT_NonCOVIDlines = trainCT_NonCOVIDfile.readlines()
    trainCT_NonCOVIDfile_path = "data\\train\\NoCOVID\\"
    for line in trainCT_NonCOVIDlines:
        if '\n' in line:
            line = line[0:-1]
        imagepath = NonCOVID_imagePath + line
        image = cv2.imread(imagepath, 0)
        image = cv2.resize(image, dsize=(256, 256))
        cv2.imwrite(trainCT_NonCOVIDfile_path + line, image)

    valCT_NonCOVIDfile = open(valCT_NonCOVID, "r")
    valCT_NonCOVIDlines = valCT_NonCOVIDfile.readlines()
    valCT_NonCOVIDfile_path = "data\\val\\NoCOVID\\"
    for line in valCT_NonCOVIDlines:
        if '\n' in line:
            line = line[0:-1]
        imagepath = NonCOVID_imagePath + line
        image = cv2.imread(imagepath, 0)
        image = cv2.resize(image, dsize=(256, 256))
        cv2.imwrite(valCT_NonCOVIDfile_path + line, image)

    testCT_COVIDfile = open(testCT_COVID, "r")
    testCT_COVIDlines = testCT_COVIDfile.readlines()
    testCT_COVIDfile_path = "data\\test\\COVID\\"
    for line in testCT_COVIDlines:
        if '\n' in line:
            line = line[0:-1]
        imagepath = COVID_imagePath + line
        image = cv2.imread(imagepath, 0)
        image = cv2.resize(image, dsize=(256, 256))
        cv2.imwrite(testCT_COVIDfile_path + line, image)

    trainCT_COVIDfile = open(trainCT_COVID, "r")
    trainCT_COVIDlines = trainCT_COVIDfile.readlines()
    trainCT_COVIDfile_path = "data\\train\\COVID\\"
    for line in trainCT_COVIDlines:
        if '\n' in line:
            line = line[0:-1]
        imagepath = COVID_imagePath + line
        image = cv2.imread(imagepath, 0)
        image = cv2.resize(image, dsize=(256, 256))
        cv2.imwrite(trainCT_COVIDfile_path + line, image)

    valCT_COVIDfile = open(valCT_COVID, "r")
    valCT_COVIDlines = valCT_COVIDfile.readlines()
    valCT_COVIDfile_path = "data\\val\\COVID\\"
    for line in valCT_COVIDlines:
        if '\n' in line:
            line = line[0:-1]
        imagepath = COVID_imagePath + line
        image = cv2.imread(imagepath, 0)
        image = cv2.resize(image, dsize=(256, 256))
        cv2.imwrite(valCT_COVIDfile_path + line, image)


def saveimage2csvfile():
    save_file2csv("data\\test\\COVID", "data\\test_COVID.csv", 1)
    save_file2csv("data\\test\\NoCOVID", "data\\test_NoCOVID.csv", 0)
    save_file2csv("data\\train\\COVID", "data\\train_COVID.csv", 1)
    save_file2csv("data\\train\\NoCOVID", "data\\train_NoCOVID.csv", 0)
    save_file2csv("data\\val\\COVID", "data\\val_COVID.csv", 1)
    save_file2csv("data\\val\\NoCOVID", "data\\val_NoCOVID.csv", 0)


def traindataAug():
    aug = DataAug2d(rotation=45, width_shift=0.02, height_shift=0.02, zoom_range=0.01)
    aug.DataAugmentation('data\\train_COVID.csv', 20, aug_path='data\\train_aug\\COVID\\')
    aug.DataAugmentation('data\\train_NoCOVID.csv', 20, aug_path='data\\train_aug\\NoCOVID\\')


def saveaugimage2csvfile():
    save_file2csv("data\\train_aug\\COVID", "data\\train_augCOVID.csv", 1)
    save_file2csv("data\\train_aug\\NoCOVID", "data\\train_augNoCOVID.csv", 0)


if __name__ == "__main__":
    # step1
    # Data_splitintotrainvaltest()
    # step2
    # saveimage2csvfile()
    # step3
    # traindataAug()
    # step4
    saveaugimage2csvfile()
