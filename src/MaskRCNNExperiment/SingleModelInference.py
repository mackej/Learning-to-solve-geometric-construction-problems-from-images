import os
import sys
import time
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import importlib.util
import mrcnn.model as modellib
from mrcnn import visualize
import enviroment_utils as env_utils
import matplotlib.pyplot as plt

class SingleModelInference:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.model_path = None
        self.image_number = 1
        self.sum_elapsed = 0
        self.elapsed_cnt = 0
        self.avg_time_between_detections = 0
        self.avg_time_between_start = None

    def detect(self, images, verbose=0, bool_masks=True):
        if self.avg_time_between_start is not None:
            end = time.time() - self.avg_time_between_start
            n = self.elapsed_cnt
            self.avg_time_between_detections = (self.avg_time_between_detections * n + end)/(n+1)
        t_start = time.time()
        res = self.model.detect(images, verbose=verbose, bool_masks=bool_masks)
        if len(images) == 1:
            res = res[0]
        t_end = time.time() - t_start
        self.sum_elapsed += t_end
        self.elapsed_cnt += 1
        self.avg_time_between_start = time.time()
        return res

    def get_average_elapsed_time(self):
        return self.sum_elapsed / self.elapsed_cnt

    @staticmethod
    def load_model(model_path, output_to=sys.stdout, batch_size=1):
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        alternative_dataset = os.path.abspath(
            ROOT_DIR + os.path.sep + os.path.dirname(model_path) + os.path.sep + "GeometryDataset.py")
        if os.path.exists(alternative_dataset):
            # if model file have its special dataset... then load it instead of default one
            spec = importlib.util.spec_from_file_location("alternative_dataset", alternative_dataset)
            ds = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ds)
            dataset = ds.GeometryDataset()
        else:
            from MaskRCNNExperiment import GeometryDataset

            dataset = GeometryDataset.GeometryDataset()
        dataset.PrepareDataGen(tool_list_file=os.path.join(ROOT_DIR, os.path.dirname(model_path)))
        # dataset.PrepareDataGen()
        dataset.prepare()

        alternative_config = os.path.abspath(
            ROOT_DIR + os.path.sep + os.path.dirname(model_path) + os.path.sep + "GeometryConfig.py")
        if os.path.exists(alternative_config):
            # if model file have its special config... then load it instead of default one
            spec = importlib.util.spec_from_file_location("alternative_config", alternative_config)
            ds = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ds)
            inference_config = ds.GeometryConfig()
        else:
            import MaskRCNNExperiment.GeometryConfig as gconfig
            inference_config = gconfig.GeometryConfig()
        #0 for the best performance wih tree search but 0.75 without tree search is enough
        inference_config.DETECTION_MIN_CONFIDENCE = 0.0
        inference_config.GPU_COUNT = 1
        inference_config.IMAGES_PER_GPU = batch_size
        inference_config.BATCH_SIZE = batch_size
        inference_config.NUM_CLASSES = dataset.get_number_od_classes()
        inference_config.reproduce()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)
        if model_path == "last":
            model_path = model.find_last()
        else:
            model_path = os.path.join(ROOT_DIR, model_path)

        # Load trained weights
        print("Loading weights from {}".format(model_path), file=output_to, flush=True)
        model.load_weights(model_path, by_name=True)
        return SingleModelInference(model, dataset, inference_config)

    def show_model_input(self, image, pred, caption_col, visualization_scale, ax):
        visualize.display_instances(image, pred['rois'], pred['masks'] > 0.5, pred['class_ids'],
                                    self.dataset.class_names, pred['scores'], caption_col=caption_col,
                                    show_mask=False, show_bbox=False,
                                    ax=ax, upscale=visualization_scale)
        #plt.show()
#        plt.savefig("input_image{}.png".format(i), dpi=1024)

    def show_model_output(self, image, pred, caption_col, visualization_scale, ax):
        colors = [(0.5, 0, 0.5), (0.5, 0.5, 0.0), (0, 0.5, 0.5), (0.5, 0, 0.25)]
        visualize.display_instances(image, pred['rois'], pred['masks'] > 0.5, pred['class_ids'],
                                    self.dataset.class_names, pred['scores'],
                                    caption_col=caption_col, upscale=visualization_scale, ax=ax, colors=colors)
        #plt.show()
        # plt.savefig("output_image{}.png".format(i), dpi=1024)


    def inference_step(self, pred, hint, m):
        return env_utils.EnvironmentUtils.execute_one_step(pred, self.dataset, hint, m)
