import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import glob
import enviroment_utils as env_utils
import time

from MaskRCNNExperiment.SingleModelInference import SingleModelInference

class MultipleModelsInference(SingleModelInference):
    def __init__(self, models):
        self.model = models
        self.enabled = [True]*len(models)
        self.avg_time_between_start_mult = None
        self.avg_time_between_detections_mult = 0
        # first detection tends to have higher evaluation time... -1 to skip it from averages
        self.elapsed_cnt_mult = -1
        self.sum_elapsed_avg = 0

    @classmethod
    def load_model_exact_path(cls, dirs, match=".*", output_to=sys.stdout, model_type=SingleModelInference,model_batch_size=1):
        models=[]
        for i in dirs:
            full_path = os.path.join(ROOT_DIR, i)
            model = model_type.load_model(full_path, output_to=output_to,batch_size=model_batch_size)
            model.model_path = full_path
            models.append(model)
        return cls(models)

    @classmethod
    def load_model(cls, dirs, match=".*", output_to=sys.stdout, model_type=SingleModelInference):
        dirs = str(dirs).split(";")
        models = []
        for level_pack_dir in dirs:
            pack_full_path = os.path.join(ROOT_DIR, level_pack_dir)
            for model_dir in os.listdir(pack_full_path):
                latest_checkpoint = max(glob.glob(pack_full_path + os.path.sep + model_dir + os.path.sep + "*.h5"))
                model = model_type.load_model(latest_checkpoint, output_to=output_to)
                models.append(model)
        return cls(models)


    def detect(self, images, verbose=0, bool_masks=True, high_level_verbose=0):
        result = []
        if self.avg_time_between_start_mult is not None and self.elapsed_cnt_mult >=0:
            end = time.time() - self.avg_time_between_start_mult
            n = self.elapsed_cnt_mult
            self.avg_time_between_detections_mult = (self.avg_time_between_detections_mult * n + end)/(n+1)
        t_start = time.time()
        for i in range(len(self.model)):
            if not self.enabled[i]:
                result.append(None)
                continue
            if high_level_verbose > 0:
                print("Prediction from {} - done.".format(self.model[i].model_path))
            pred = self.model[i].detect(images, verbose=verbose, bool_masks=bool_masks)
            result.append(pred)
        t_end = time.time() - t_start
        if self.elapsed_cnt_mult >=0:
            self.sum_elapsed_avg = (self.sum_elapsed_avg * self.elapsed_cnt_mult + t_end) / (self.elapsed_cnt_mult +1)
        self.elapsed_cnt_mult += 1
        self.avg_time_between_start_mult = time.time()
        #print("avg time for detection = {}, avg time between two detections {}".format(self.sum_elapsed_avg,self.avg_time_between_detections_mult))
        return result

    def get_average_elapsed_time(self):
        sum = 0
        for i in self.model:
            sum += i.get_average_elapsed_time
        return sum / len(self.model)

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


