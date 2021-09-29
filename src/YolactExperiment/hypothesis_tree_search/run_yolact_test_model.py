import sys
import os
import argparse
import random
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from HypothesisExplorer.HypothesisTreeSearch import *
import HypothesisExplorer.models_config as config
from YolactExperiment.YolactInference_SingleModel import YolactSingleModelInference as yolact_model
from LevelSelector import *

random.seed(42)
np.random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Yolact Training Script')
    parser.add_argument('--log_folder', default="weights/LR_test_4_alpha_2021_06_10/Euclidea_79_1000000.pth", type=str,
                        help='folder with logs, or multiple folder separated with comma')
    parser.add_argument('--generate_levels', default="alpha.*", type=str,
                        help='which levels should be generated')
    args = parser.parse_args()

    levels = LevelSelector.get_levels(match=args.generate_levels)
    models = [args.log_folder]

    t = HypohesisTreeSearch(models, levels, epoch_each_level=100, model_disabling=False, model_type=yolact_model)
    print(t.test_all_levels())