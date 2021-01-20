import os
import glob
from pathlib import Path
#do not show print tf logs. '0' full printing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import pickle
import shutil
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
import MaskRCNNExperiment.GeometryConfig as gconfig
from MaskRCNNExperiment.TestModel import TestModel
from MaskRCNNExperiment.DataGenerator import generate_geometry_episodes
from MaskRCNNExperiment.GeometryDataset_on_the_fly_gen import GeometryDataset_on_the_fly_gen
import mrcnn.model as modellib
import tensorflow as tf
import TrainingUtils
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import random


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_weights", default="coco", type=str,
                        help="which weight should be used "
                             "'coco' for weight from coco and "
                             "'last' for last trained model")
    parser.add_argument("--head_epochs", default=60, type=int, help="epochs for head layer training")
    parser.add_argument("--four_plus_epochs", default=120, type=int, help="epochs for 4+ layer training")
    parser.add_argument("--all_epochs", default=200, type=int, help="epochs for training the whole network")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use")
    # Data generation options - create new dataset
    parser.add_argument("--train_epochs", default=100, type=int,
                        help="Size of how many epochs are generated for training data")
    parser.add_argument("--val_epochs", default=100, type=int,
                        help="Size of how many epochs are generated for validation data")
    parser.add_argument("--history_size", default=1, type=int,
                        help="how many steps of history for each image")
    parser.add_argument("--generate_levels", default="04.*10", type=str,
                        help="regex that matches lvl names")
    parser.add_argument("--use_heat_map", default=0, type=int,
                        help="whether to use heat map or not as target, also config minimak have to be set to False")
    parser.add_argument("--mask_size", default=5, type=int,
                        help="size of target mask")
    parser.add_argument("--heat_map_covariance", default=100, type=float,
                        help="covariance of normal distribution for heat maps")
    parser.add_argument("--tool_set", default="min_by_construction", type=str,
                        help="\"min_by_levels\" to generate minimal set of tools given by level;"
                             "\"min_by_construction\" to generate minimal set of tools given by construction;"
                             " Other values for all tools")
    parser.add_argument("--visualize", default=0, type=int,
                        help="show image of environment for each step. This takes a long time,"
                             " so its for debugging purpose only.")
    # Data generation options - load existing dataset
    parser.add_argument("--load_gen_from_file", default=0, type=int,
                        help="1 to load dataset prom pickled file. 0 to generate before run")
    parser.add_argument("--train_datagen", default="datagen_train_data", type=str,
                        help="file with lzma pickled train genreator")
    parser.add_argument("--val_datagen", default="datagen_val_data", type=str,
                        help="file with lzma pickled val genreator")
    parser.add_argument("--log_folder_name", default="delete_this", type=str,
                        help="name for log folder. None to generate new unique name")
    parser.add_argument("--data_generation_style", default="on-the-fly", type=str,
                        help="'pre-sampled' or 'on-the-fly' methods")
    parser.add_argument("--seed", default=3911, type=int,
                        help="random seed for data generation.")
    args = parser.parse_args()

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    if args.load_gen_from_file > 0:
        # Loading train datagen
        with open(args.train_datagen, "rb") as data_gen_file:
            dataset_train = pickle.load(data_gen_file)
        # Loading validation datagen
        with open(args.val_datagen, "rb") as data_gen_file:
            dataset_val = pickle.load(data_gen_file)
    else:
        if args.data_generation_style == "on-the-fly":
            dataset_train = GeometryDataset_on_the_fly_gen(args)
            dataset_val = GeometryDataset_on_the_fly_gen(args)
        else:
            dataset_train, dataset_val = generate_geometry_episodes(args)
    # Directory to save logs and trained model
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
        model.model_dir = os.path.join(MODEL_DIR, args.log_folder_name)
        model.load_weights(model.find_last(), by_name=True)
    elif os.path.exists(os.path.join(MODEL_DIR, args.use_weights)):
        model.load_weights(os.path.join(MODEL_DIR, args.use_weights), by_name=True)
        model.epoch = int(args.use_weights[-7:-3])

    else:
        raise Exception('parameter --use_weights got "{}", expected "coco" or "last"'.format(args.use_weights))

    if args.log_folder_name is not None:
        model.log_dir = str(Path(model.log_dir).parent) + os.path.sep + args.log_folder_name
        check_point = model.checkpoint_path.split(os.path.sep)
        check_point[-2] = args.log_folder_name
        model.checkpoint_path = os.path.sep.join(check_point)

    Path(model.log_dir).mkdir(parents=True, exist_ok=True)
    with open(model.log_dir + os.path.sep + "_metaparams", 'w') as file:
        file.write(config.display())
        file.write(str(args))
    shutil.copy("GeometryConfig.py", model.log_dir + os.path.sep+"GeometryConfig.py")
    shutil.copy("GeometryDataset.py", model.log_dir + os.path.sep + "GeometryDataset.py")
    shutil.copy("TrainGeometryDataset.py", model.log_dir + os.path.sep + "TrainGeometryDataset.py")
    dataset_train.dump_class_info(model.log_dir)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.head_epochs,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.four_plus_epochs,
                layers='4+',
                )

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=args.all_epochs,
                layers="all")


    # plot simple loss
    TrainingUtils.events_to_pyplot(model.log_dir)

    if args.data_generation_style == "on-the-fly":
        with open(model.log_dir + os.path.sep + "on_the_fly_stats", 'w') as file:
            file.write(dataset_train.get_statistics())

    # Test model
    latest_checkpoint = max(glob.glob(model.log_dir + os.path.sep + "*.h5"), key=os.path.getctime)
    TestModel(args.generate_levels, latest_checkpoint, 500, args.history_size, 2,
              use_hint=False, visualization=False, white_visualization=False,
              log_fails=True, log_failed_levels_file=model.log_dir + os.path.sep + "unfinished_levels",
              output_to=open(model.log_dir+os.path.sep + "output.txt", 'w'),
              model_type="SingleModelInference")






