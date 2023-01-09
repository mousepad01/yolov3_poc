import numpy as np

'''
    Various constants needed in whatever places
'''

'''
    ############################################ CACHE AND DATA LOADING ############################################
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

VALIDATION_STATS_PATH = "./validation_stats/"
'''
    (relative) path for storing validation statistics
'''

TMP_CACHE_KEY = "tmp"
'''
    key for temporary cache, if needed
'''

PERMANENT_DATA_ENTRIES = 1000000
'''
    how many data entries (imgs with their gts) to permanently store in memory (and be loaded only once)
'''

PERMANENT_PRETRAIN_DATA_ENTRIES = 100000000
'''
    how many bounding box entries to permanently store in memory (and be loaded only once)
    * FOR PRETRAIN PHASE ONLY
'''

COMPRESS_GT_CACHE_LEVEL = 1
'''
    * the level of compression of GT cache batches
    * corresponds to zlib.compress levels
    * applicable only if there are GT batches cached (when PERMANENT_GT_BATCHES > 0)
    * usually 1 is good enough, because it compresses all the sparseness
'''

COMPRESS_DATA_CACHE = True
'''
    * whether to store image cache as jpg or bitmap in rap
'''

'''
    ############################################ MODEL CONSTANTS ############################################
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

PRETRAIN_BOX_SIZE = (128, 128)
'''
    size for input in classification pre-training of backbone
'''
assert(PRETRAIN_BOX_SIZE <= IMG_SIZE)

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

IGNORED_IOU_THRESHOLD = 0.5
'''
    min IOU for non-assigned anchors or predictions, for them to be completely ignored in loss function
'''

'''
    ############################################ OUTPUT ############################################
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

SHOW_RESIZE_FACTOR = 2.3
'''
    resize factor for showing images (compared to IMG_SIZE)
'''

'''
    ############################################ RANDOMNESS ############################################
'''

NOISE_MASK_POOL_LEN = 1000
'''
    how many noise masks for image augmentation to precompute
'''

#FIXME
NOISE_MASK_POOL = [np.random.uniform(-20, 20, (IMG_SIZE[0], IMG_SIZE[1], 3)) for _ in range(NOISE_MASK_POOL_LEN)]
'''
    precomputed noise masks for image augmentation
'''