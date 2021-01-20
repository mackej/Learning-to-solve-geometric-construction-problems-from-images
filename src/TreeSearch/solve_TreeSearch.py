import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from LevelSelector import LevelSelector

import matplotlib.pyplot as plt
from TreeSearch import Deepening
import mrcnn.model as modellib
from mrcnn import visualize
from GeometryObjectsDetector import DetectorGeometryDataset
import GeometryObjectsDetector.DetectorGeometryConfig as gconfig
from py_euclidea import multi_level
import TreeSearch.Deepening
import TreeSearch.IterativeDeepening
import TreeSearch.IterativeDeepeningWithMRCNNmodel as ITD_mrcnn
import random
import numpy as np

if __name__ == "__main__":
    import argparse
    seed = 42
    np.random.seed(seed=seed)
    random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_levels", default="alpha.*", type=str,
                        help="regex that matches lvl names")
    parser.add_argument("--model_path",
                        default="logs_detector/detector01/mask_rcnn_detector_of_geom_primitives_0140.h5", type=str,
                        help="loads model. 'last' for last trained model in logs")
    parser.add_argument("--episodes", default=1, type=int,
                        help="how many episodes are used for evaluation for each environment.")

    parser.add_argument("--mrcnn_solver_model", default="logs/1_step_history_test02/mask_rcnn_geometryfromimages_0180.h5", type=str,
                        help="how many episodes are used for evaluation for each environment.")

    args = parser.parse_args()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    class InferenceConfig(gconfig.GeometryConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    def get_ax(rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(800/90, 800/90))
        return ax


    dataset = DetectorGeometryDataset.DetectorGeometry()
    dataset.PrepareDataGen(history_size=1, )
    dataset.prepare()

    inference_config = InferenceConfig()
    inference_config.IMAGE_META_SIZE += -inference_config.NUM_CLASSES + dataset.get_number_od_classes()
    inference_config.NUM_CLASSES = dataset.get_number_od_classes()

    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    if args.model_path == "last":
        model_path = model.find_last()
    else:
        model_path = os.path.join(ROOT_DIR, args.model_path)

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    levels = LevelSelector.get_levels(match=args.generate_levels)
    m = multi_level.MultiLevel((
        levels
    ))
    ItD = TreeSearch.IterativeDeepening.IterativeDeepening(tool_hints=False)
    #ItD = ITD_mrcnn.ITD_with_MRCNN_model(args.mrcnn_solver_model, starting_depth=5,tool_hints=False)

    for level in range(len(levels)):
        for e in range(args.episodes):
            m.next_level(level)
            #print(levels[level])
            ItD.deepening(m, dataset, model, levels[level][1],
                          starting_depth=len(m.cur_env.construction), max_depth=len(m.cur_env.construction))

