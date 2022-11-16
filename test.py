import numpy as np
import tensorflow as tf
import cv2 as cv

from data_processing import *

def main():

    DPATH = "./data/val2017/"

    # 182611, 243204
    
    data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH, train_info_path=DataManager.VALIDATION_INFO_PATH)
    data_manager.load_info()

    k = 0
    for img_id, props in data_manager.imgs["validation"].items():
        
        if k == 0:
            id0, prop0 = img_id, props
            k += 1
        
        elif k == 1:
            id1, prop1 = img_id, props
            k += 1

        else:
            break

    im0 = cv.imread(DPATH + prop0["filename"])
    im1 = cv.imread(DPATH + prop1["filename"])

    cv.imshow(f"{id0}", data_manager.resize_with_pad(im0)[0])
    cv.waitKey(0)

    cv.imshow(f"{id1}", data_manager.resize_with_pad(im1)[0])
    cv.waitKey(0)

    return

    print(im0.shape)

    for bbox_d in prop1["objs"]:

        y1, x1, h, w = bbox_d["bbox"]
        x1, y1, w, h = np.int32(np.floor(x1)), np.int32(np.floor(y1)), np.int32(np.floor(w)), np.int32(np.floor(h))

        cv.rectangle(im1, (y1, x1), (y1 + h, x1 + w), color=(255, 0, 0), thickness=2)

        im1[x1][y1] = (0, 0, 255)
        im1[x1 + w][y1 + h] = (0, 0, 255)

        #for i in range()

    cv.imshow(f"{id0}", im0)
    cv.waitKey(0)
    
    #return 

    cv.imshow(f"{id1}", im1)
    cv.waitKey(0)
    
def jmain():

    a = np.random.uniform(0, 1, (133, 200, 3))
    b = np.zeros((200 - 133, 200, 3))
    b = np.concatenate([a, b], 0)

    print(a.shape, b.shape)

    a = np.random.uniform(0, 1, (200, 133, 3))
    b = np.zeros((200, 200 - 133, 3))
    b = np.concatenate([a, b], 1)

    print(a.shape, b.shape)

main()