import cv2 as cv

'''
    Various constants needed in whatever places
'''

DATA_LOAD_BATCH_SIZE = 32
'''
    batch size just for loading
'''

IMG_SIZE = (416, 416)
'''
    fixed image input size
'''

GRID_CELL_CNT = [13, 26, 52]
'''
    for each scale, the value of S
'''

SCALE_CNT = 3
'''
    should be kept fixed, defined as a kind of macro
'''

ANCHOR_PERSCALE_CNT = 3
'''
    numbers of anchor types per scale (A dimension)
'''

CLASS_TO_COLOR = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255), (165, 42, 42), (255, 140, 0), (255, 255, 255)]
'''
    class one hot encoding idx to color
'''
for idx, rgb in enumerate(CLASS_TO_COLOR):
    CLASS_TO_COLOR[idx] = (rgb[2], rgb[1], rgb[0])
