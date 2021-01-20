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
    models = [models[-1], models[-13]]
    levels = [levels[-1], levels[-13]]
    models = config.avaliable_models['Alpha_One_by_One'].values()
    levels = LevelSelector.get_levels(match="alpha.*")
    t = HypohesisTreeSearch(models, levels, epoch_each_level=5)
    print(t.test_all_levels())