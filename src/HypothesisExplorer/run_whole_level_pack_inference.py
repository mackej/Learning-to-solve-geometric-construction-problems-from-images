import sys
import os
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from HypothesisExplorer.HypothesisTreeSearch import *
import HypothesisExplorer.models_config as config
from LevelSelector import *

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # logs/Beta_as_whole;logs/Gamma_as_whole;logs/redoing_alpha
    parser.add_argument("--pack", default="alpha",
                        type=str,
                        help="specify level pack")
    args = parser.parse_args()
    models = config.get_one_by_one_models(config.avaliable_models)
    levels = LevelSelector.get_levels()
    models = config.avaliable_models['levels_as_whole']
    leave_out = args.pack
    del models['everything']
    del models[leave_out]
    levels = LevelSelector.get_levels(match=leave_out+".*")
    t = HypohesisTreeSearch(models.values(), levels, epoch_each_level=1, model_disabling=False)
    print(t.test_all_levels())