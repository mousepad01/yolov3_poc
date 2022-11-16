import numpy as np
import tensorflow as tf
import cv2 as cv

from data_processing import *
from anchor_kmeans import *
from model import *

def main():
    
    #FIXME
    data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH)
    data_manager.load_info()
    
    # test loading images
    '''for b in data_manager.load_images("train", False):
        
        cv.imshow("s", b[0])
        cv.waitKey(100)'''

    # test loss
    #yolov3_loss(None, None)

if __name__ == "__main__":
    main()