
'''
    Various constants needed in whatever places
'''

DATA_LOAD_BATCH_SIZE = 128
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