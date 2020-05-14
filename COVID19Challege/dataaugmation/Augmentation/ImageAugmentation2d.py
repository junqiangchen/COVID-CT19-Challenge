from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2


class DataAug2d(object):
    '''
    aug 2d image
    '''

    def __init__(self, rotation=5, width_shift=0.01, height_shift=0.01, zoom_range=0.01,
                 rescale=1.1, horizontal_flip=True, vertical_flip=False):
        # define data preparation
        self.__datagen = ImageDataGenerator(rotation_range=rotation, width_shift_range=width_shift,
                                            height_shift_range=height_shift,
                                            zoom_range=zoom_range,
                                            rescale=rescale, horizontal_flip=horizontal_flip,
                                            vertical_flip=vertical_flip,
                                            fill_mode='nearest')

    def __ImageMaskTranform(self, images_path, index, number):
        # reshape to be [samples][width][height][channels]
        imagesample = cv2.imread(images_path, 0)
        srcimage = imagesample.reshape([1, imagesample.shape[0], imagesample.shape[1], 1])
        i = 0
        for batch1 in self.__datagen.flow(srcimage):
            i += 1
            batch1 = batch1[0, :, :, :]
            for j in range(batch1.shape[2]):
                augfile_path = self.aug_path + str(index) + '_' + str(i) + ".jpg"
                batchx = batch1.reshape([imagesample.shape[0], imagesample.shape[1]])
                cv2.imwrite(augfile_path, batchx)
            if i > number - 1:
                break

    def DataAugmentation(self, filepath, number=20, aug_path=None):
        csvdata = pd.read_csv(filepath)
        data = csvdata.iloc[:, 1].values
        self.aug_path = aug_path
        for index in range(data.shape[0]):
            # For images
            images_path = data[index]
            self.__ImageMaskTranform(images_path, index, number)
