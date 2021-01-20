import sys
import os
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from HypothesisExplorer.HypothesisTreeSearch import *
import HypothesisExplorer.models_config as config
from LevelSelector import *

if __name__ == '__main__':
    models = config.get_one_by_one_models(config.avaliable_models)
    levels = LevelSelector.get_levels()
    models = [config.avaliable_models['levels_as_whole']['everything']]
    levels = LevelSelector.get_levels(match=".*")
    t = HypohesisTreeSearch(models, levels, epoch_each_level=1, model_disabling=False)
    print(t.test_all_levels())