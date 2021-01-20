import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from GeometryObjectsDetector.DataGenerator import generate_geometry_episodes

import pickle
import lzma

if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_epochs", default=20_000, type=int,
                        help="Size of how many epochs are generated for training")
    parser.add_argument("--val_epochs", default=1000, type=int,
                        help="Size of how many epochs are generated for validation")
    parser.add_argument("--history_size", default=1, type=int,
                        help="how many steps of history for each image")
    parser.add_argument("--generate_levels", default="04.*06", type=str,
                        help="regex that matches lvl names")

    parser.add_argument("--export_train_file", default="datagen_train_data", type=str,
                        help="output file for train generator")
    parser.add_argument("--export_val_file", default="datagen_val_data", type=str,
                        help="output file for validation generator")
    parser.add_argument("--mask_size", default=5, type=float,
                        help="size of target mask")
    parser.add_argument("--visualize", default=1, type=int,
                        help="show image of environment for each step. This takes a long time,"
                             " so its for debugging purpose only.")
    args = parser.parse_args()

    data_gen, data_gen_val = generate_geometry_episodes(args)

    with open(args.export_train_file, "wb") as data_gen_file:
        pickle.dump(data_gen, data_gen_file)
    with open(args.export_val_file, "wb") as data_gen_file:
        pickle.dump(data_gen_val, data_gen_file)




