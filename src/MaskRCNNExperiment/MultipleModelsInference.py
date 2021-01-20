import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import glob
import enviroment_utils as env_utils

from MaskRCNNExperiment.SingleModelInference import SingleModelInference

class MultipleModelsInference(SingleModelInference):
    def __init__(self, models):
        self.model = models
        self.enabled = [True]*len(models)
    @classmethod
    def load_model_exact_path(cls, dirs, match=".*", output_to=sys.stdout):
        models=[]
        for i in dirs:
            full_path = os.path.join(ROOT_DIR, i)
            model = SingleModelInference.load_model(full_path, output_to=output_to)
            model.model_path = full_path
            models.append(model)
        return cls(models)

    @classmethod
    def load_model(cls, dirs, match=".*", output_to=sys.stdout):
        dirs = str(dirs).split(";")
        models = []
        for level_pack_dir in dirs:
            pack_full_path = os.path.join(ROOT_DIR, level_pack_dir)
            for model_dir in os.listdir(pack_full_path):
                latest_checkpoint = max(glob.glob(pack_full_path + os.path.sep + model_dir + os.path.sep + "*.h5"))
                model = SingleModelInference.load_model(latest_checkpoint, output_to=output_to)
                models.append(model)
        return cls(models)


    def detect(self, images, verbose=0, bool_masks=True, high_level_verbose=0):
        result = []

        for i in range(len(self.model)):
            if not self.enabled[i]:
                result.append(None)
                continue
            if high_level_verbose > 0:
                print("Prediction from {} - done.".format(self.model[i].model_path))
            pred = self.model[i].detect(images, verbose=verbose, bool_masks=bool_masks)
            result.append(pred)
        return result

    def show_model_input(self, image, pred, caption_col, visualization_scale, ax):
        self.model[0].show_model_input(image, pred[0], caption_col, visualization_scale, ax)

    def show_model_output(self, image, pred, caption_col, visualization_scale, ax):
        assert len(pred) == len(self.model)
        for i in range(len(self.model)):
            self.model[i].show_model_output(image, pred[i], caption_col, visualization_scale, ax)

    def inference_step(self, pred, hint, multi_level):
        raise NotImplemented
        assert len(pred) == len(self.model)
        multi_level.cur_env.get_construction_steps()
        all_hypothesis = []
        for i in range(len(pred)):
            p = pred[i]
            m = self.model[i]
            h = env_utils.EnvironmentUtils.prepare_all_hypothesis(p, m.dataset, multi_level)
            all_hypothesis.append(h)


