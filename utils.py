
'''
    Various constants needed in whatever places
'''

DATA_CACHE_PATH = "./data_cache/"
'''
    (relative) path for storing data (anchors, gt masks) caches
'''

MODEL_CACHE_PATH = "./saved_models/"
'''
    (relative) path for storing saved models
'''

TRAIN_STATS_PATH = "./train_stats/"
'''
    (relative) path for storing train statistics
'''

TMP_CACHE_KEY = "tmp"
'''
    key for temporary cache, if needed
'''

DATA_LOAD_BATCH_SIZE = 128
'''
    batch size just for loading
'''

GT_LOAD_BATCH_SIZE = 1024
'''
    batch size just for loading ground truth (bool masks, target masks)
'''
assert(GT_LOAD_BATCH_SIZE % DATA_LOAD_BATCH_SIZE == 0)
assert(GT_LOAD_BATCH_SIZE >= DATA_LOAD_BATCH_SIZE)

PERMANENT_DATA_BATCHES = 10000000000
'''
    how many data batches to permanently store in memory (and be loaded only once)
'''

PERMANENT_DATA_ENTRIES = PERMANENT_DATA_BATCHES * DATA_LOAD_BATCH_SIZE
'''
    how many data entries to permanently store in memory (and be loaded only once)
'''

PERMANENT_GT_BATCHES = 10000000000
'''
    how many gt entries to permanently store in memory (and be loaded only once)
'''

IMG_SIZE = (416, 416)
'''
    fixed image input size
'''

MIN_BBOX_DIM = 3
'''
    minimum bounding box dimension (w or h, in pixes, after resize to IMG_SIZE)
    to filter ground truth
'''

CLASSIFICATION_SIZE = (256, 256)
'''
    (UNUSED)
    classification pre-training of backbone
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

CLASS_TO_COLOR = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255), (165, 42, 42), (255, 140, 0), (255, 255, 255)] * 10
'''
    class one hot encoding idx to color
'''
for idx, rgb in enumerate(CLASS_TO_COLOR):
    CLASS_TO_COLOR[idx] = (rgb[2], rgb[1], rgb[0])

LOSS_OUTPUT_PRECISION = 4
'''
    how many decimals for loss output - does not influence in any way the model
'''

TRAIN_DATA_PATH = "./data/train2017/"
'''
    train data path
'''

VALIDATION_DATA_PATH = "./data/val2017/"
'''
    validation data path
'''

TRAIN_INFO_PATH = "./data/annotations/instances_train2017.json"
'''
    train metadata
'''

VALIDATION_INFO_PATH = "./data/annotations/instances_val2017.json"
'''
    validation metadata
'''
