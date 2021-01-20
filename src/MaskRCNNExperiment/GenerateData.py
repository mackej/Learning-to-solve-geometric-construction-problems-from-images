import os
import psutil
import sys
import numpy as np
import random
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from MaskRCNNExperiment.DataGenerator import generate_geometry_episodes

import pickle
import lzma

if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_epochs", default=20, type=int,
                        help="Size of how many epochs are generated for training")
    parser.add_argument("--val_epochs", default=10, type=int,
                        help="Size of how many epochs are generated for validation")
    parser.add_argument("--history_size", default=1, type=int,
                        help="how many steps of history for each image")
    parser.add_argument("--generate_levels", default="02.*02", type=str,
                        help="regex that matches lvl names")

    parser.add_argument("--export_train_file", default="datagen_train_data", type=str,
                        help="output file for train generator")
    parser.add_argument("--export_val_file", default="datagen_val_data", type=str,
                        help="output file for validation generator")
    parser.add_argument("--use_heat_map", default=False, type=bool,
                        help="whether to use heat map or not as target, also config minimak have to be set to False")
    parser.add_argument("--heat_map_covariance", default=100, type=float,
                        help="covariance of normal distribution for heat maps")
    parser.add_argument("--mask_size", default=5, type=float,
                        help="size of target mask")
    parser.add_argument("--visualize", default=1, type=int,
                        help="show image of environment for each step. This takes a long time,"
                             " so its for debugging purpose only.")
    parser.add_argument("--tool_set", default="min_by_construction", type=str,
                        help="\"min_by_levels\" to generate minimal set of tools given by level;"
                             "\"min_by_construction\" to generate minimal set of tools given by construction;"
                             " Other values for all tools")
    parser.add_argument("--seed", default=3911, type=int,
                        help="random seed for data generation.")
    args = parser.parse_args()

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    data_gen, data_gen_val = generate_geometry_episodes(args)
    print('memory use:', psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30)
    with open(args.export_train_file, "wb") as data_gen_file:
        pickle.dump(data_gen, data_gen_file)
    with open(args.export_val_file, "wb") as data_gen_file:
        pickle.dump(data_gen_val, data_gen_file)




