import os
from pathlib import Path
#do not show print tf logs. '0' full printing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import pickle
import lzma
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.utils as utils
import GeometryObjectsDetector.DetectorGeometryConfig as gconfig
from GeometryObjectsDetector.DataGenerator import generate_geometry_episodes
import mrcnn.model as modellib
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_weights", default="coco", type=str,
                        help="which weight should be used "
                             "'coco' for weight from coco and "
                             "'last' for last trained model")
    parser.add_argument("--head_epochs", default=50, type=int, help="epochs for head layer training")
    parser.add_argument("--all_epochs", default=100, type=int, help="epochs for training model")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use")
    # Data generation options - create new dataset
    parser.add_argument("--train_epochs", default=100, type=int,
                        help="Size of how many epochs are generated for training data")
    parser.add_argument("--val_epochs", default=100, type=int,
                        help="Size of how many epochs are generated for validation data")
    parser.add_argument("--history_size", default=1, type=int,
                        help="how many steps of history for each image")
    parser.add_argument("--generate_levels", default="[alpha|beta]", type=str,
                        help="regex that matches lvl names")
    parser.add_argument("--mask_size", default=5, type=int,
                        help="size of target mask")
    parser.add_argument("--visualize", default=0, type=int,
                        help="show image of environment for each step. This takes a long time,"
                             " so its for debugging purpose only.")
    args = parser.parse_args()

    dataset_train, dataset_val = generate_geometry_episodes(args)
    # Directory to save logs and trained model
    for i in range(50):
        m  = dataset_train.load_mask(i)
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = gconfig.GeometryConfig()
    config.IMAGE_META_SIZE += -config.NUM_CLASSES + dataset_train.get_number_od_classes()
    config.NUM_CLASSES = dataset_train.get_number_od_classes()
    config.GPU_COUNT = args.gpus
    config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT



    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # use coco weights to start with

    if args.use_weights == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    elif args.use_weights == "last":
        model.load_weights(model.find_last(), by_name=True)
    else:
        raise Exception('parameter --use_weights got "{}", expected "coco" or "last"'.format(args.use_weights))

    Path(model.log_dir).mkdir(parents=True, exist_ok=True)
    with open(model.log_dir + os.path.sep + "_metaparams", 'w') as file:
        file.write(config.display())
        file.write(str(args))

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.head_epochs,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=args.all_epochs,
                layers="all")





