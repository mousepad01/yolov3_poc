import numpy as np
import tensorflow as tf

from data_processing import DataManager
from anchor_kmeans import AnchorFinder

def main():
    
    #FIXME
    data_manager = DataManager(train_data_path=DataManager.VALIDATION_DATA_PATH)
    data_manager.load_info()

if __name__ == "__main__":
    main()