import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import mrcnn.utils as utils
from collections import namedtuple


import matplotlib.pyplot as plt
from PIL import Image
from LevelSelector import LevelSelector
from collections import deque
import pickle
import matplotlib.pyplot as plt
from mrcnn import visualize
from GeometryObjectsDetector import DetectorGeometryDataset
from GeometryObjectsDetector.DetectorUtils import *
import GeometryObjectsDetector.DetectorGeometryConfig as gconfig
import mrcnn.model as modellib
from py_euclidea import multi_level
import numpy as np
from enviroment_utils import EnvironmentUtils
from py_euclidea.constructions import *
from py_euclidea.tools import *
from py_euclidea.environment import ConstrStep, MovableStep

class ReinforceEvironment:
    """
    api for reinforce learning.
    """
    def __init__(self, args):
        self.tool_list = [
            #(PointTool(), 'Point'),
            (StdTool(line_tool, (Point, Point), Line), 'Line'),
            (StdTool(circle_tool, (Point, Point), Circle), 'Circle'),
            (StdTool(perp_bisector_tool, (Point, Point), Line), 'Perpendicular_Bisector'),
            (StdTool(angle_bisector_tool, (Point, Point, Point), Line), 'Angle_Bisector'),
            (StdTool(perp_tool, (Line, Point), Line), 'Perpendicular'),
            (StdTool(parallel_tool, (Line, Point), Line), 'Parallel'),
            (StdTool(compass_tool, (Point, Point, Point), Circle), 'Compass'),
            (IntersectionTool(), 'intersection')]



        # TODO: model can produce more than maximal number of actions....
        #  but it can be later on restricted inside model config
        # maximum number of action to choose from
        self.max_number_of_actions = args.maximal_action_space
        self.maximum_moves_inside_environment = 10
        self.total_number_of_tools = len(self.tool_list)

        # after self.max_idle_time's iteration when nothing is constructed task is terminated with penalty
        self.max_idle_time = 3
        self.no_action_repeats = 0

        self.statistic_period = 1
        self.mean_reward_q = deque(maxlen=self.statistic_period)

        self.step_in_episodes = 0
        self.episode = 0
        self.last_prediction = None

        self.dataset = DetectorGeometryDataset.DetectorGeometry()
        self.dataset.PrepareDataGen(history_size=args.history_size)
        self.dataset.prepare()

        levels = LevelSelector.get_levels(match=args.generate_levels)
        self.multilevel = multi_level.MultiLevel(levels)
        self.tool_list_reindex = [self.multilevel.tool_name_to_index[t[1]] for t in self.tool_list]

        model_dir = os.path.join(ROOT_DIR, "logs")
        inference_config = InferenceConfig()
        inference_config.IMAGE_META_SIZE += -inference_config.NUM_CLASSES + self.dataset.get_number_od_classes()
        inference_config.NUM_CLASSES = self.dataset.get_number_od_classes()
        self.mrcnn_model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=model_dir)
        if args.detector_model == "last":
            model_path = self.mrcnn_model.find_last()
        else:
            model_path = os.path.join(ROOT_DIR, args.detector_model)

        # Load trained weights
        print("Loading weights from ", model_path)
        self.mrcnn_model.load_weights(model_path, by_name=True)

        self.history_size = args.history_size
        self.history = deque(maxlen=self.history_size)
        self.use_hint = args.hint > 0
        self.additional_moves = args.additional_moves

        self.image_and_masks_shape = [self.multilevel.out_size[0], self.multilevel.out_size[1], self.max_number_of_actions]
        self.class_ids_shape = [self.max_number_of_actions]
        self.tool_mask_shape = [len(self.tool_list)]

        self.episode_sum_reward = None


    def build_state(self, pred):
        """
        builds features of image based on mrcnn model and then everything puts together as one state for
        reinforce learning
        :return:
        """
        tool_mask = self.get_tool_mask()

        pred_size = len(pred['scores'])

        adjusted_scores = np.zeros((self.max_number_of_actions))
        adjusted_scores[:pred_size] = pred['scores']

        adjusted_class_ids = np.zeros((self.max_number_of_actions)) - 1
        adjusted_class_ids[:pred_size] = pred['class_ids']

        adjusted_masks = np.zeros((self.multilevel.out_size[0], self.multilevel.out_size[1], self.max_number_of_actions))
        adjusted_masks[:, :, :pred_size] = pred['masks']
        return {
            "image_and_mask": [adjusted_masks],
            "class_ids": [adjusted_class_ids],
            "tool_mask": [tool_mask]
        }



    def apply_action(self, tool, clicks):
        #clicks = [0, 1, 2]
        #tool = 2
        # Action click is empty. Return bad reward to punish this move. Those move does not generate any move.
        if max(clicks) >= len(self.last_prediction["class_ids"]):
            return -1000, False
        # Tool is not allowed in this level
        if not self.get_tool_mask()[tool]:
            return -100, False


        classes = [self.last_prediction["class_ids"][c] for c in clicks]
        if max(classes) > 3:
            return -50, False
        t = self.tool_list[tool][0]

        arg_objects = []
        if isinstance(t, StdTool):
            for i in range(len(t.in_types)):
                type = t.in_types[i]
                mask_of_object = self.last_prediction["masks"][:, :, clicks[i]]
                multi_level_object = DetectorUtils.match_prediction(mask_of_object, classes[i], self.dataset, self.multilevel)

                if type == Point and not isinstance(multi_level_object, Point):
                    arg_objects.append(self.reasonable_random_point_on(multi_level_object).index)
                else:
                    arg_objects.append(multi_level_object.index)
        tool_status = t.run_known_objects(self.multilevel.cur_env, arg_objects)
        if tool_status is False:
            return -10, False
        r, done = self.multilevel.evaluate_last_step()
        return r, done



    def step(self, tool, clicks):
        r, done = self.apply_action(tool, clicks)

        image = EnvironmentUtils.build_image_from_multilevel(self.multilevel, self.history)
        #Image.fromarray(image, 'RGB').show()
        self.last_prediction = self.mrcnn_model.detect([image], verbose=0, bool_masks=False)[0]
        self.sort_predictions()

        last_state = image[:, :, 0]
        self.history.append(last_state)
        self.step_in_episodes += 1

        # end early .... too many steps were done
        if self.step_in_episodes > self.multilevel.get_construction_length() + self.additional_moves:
            done = True

        self.episode_sum_reward += r
        if self.step_in_episodes >= self.maximum_moves_inside_environment:
            done = True
        return self.build_state(self.last_prediction), r, done

    def reset(self):
        self.episode +=1
        if self.episode_sum_reward is not None:
            self.mean_reward_q.append(self.episode_sum_reward)
        self.episode_sum_reward = 0
        self.step_in_episodes = 0
        self.no_action_repeats = 0
        self.multilevel.next_level()
        for i in range(self.history_size):
            self.history.append(np.zeros(self.multilevel.out_size))
        image = EnvironmentUtils.build_image_from_multilevel(self.multilevel, self.history)
        #Image.fromarray(image, 'RGB').show()
        last_state = image[:, :, 0]
        self.history.append(last_state)

        self.last_prediction = self.mrcnn_model.detect([image], verbose=0, bool_masks=False)[0]
        self.sort_predictions()

        if self.episode % self.statistic_period == 0:
            print('Mean reward at episode:', self.episode, ': ', np.mean(np.array(self.mean_reward_q)))
        return self.build_state(self.last_prediction)

    def sort_predictions(self):
        indices = np.argsort(self.last_prediction["class_ids"])
        for k in self.last_prediction.keys():
            if k == "masks":
                self.last_prediction[k] = self.last_prediction[k][:, :, indices]
            else:
                self.last_prediction[k] = self.last_prediction[k][indices]

    def reasonable_random_point_on(self, ob):
        # TODO: improve random object genertation
        while True:
            if isinstance(ob, Circle):
                p = ob.r * unit_vector(np.random.random() * 2 * np.pi) + ob.c
            if isinstance(ob, Line):
                offset = 0.1
                coef = np.random.uniform(offset, 1 - offset)
                x, y = ob.get_endpoints(self.multilevel.corners)
                p = coef * x + (1 - coef) * y

            scaled_coors = p / self.multilevel.scale
            if np.all(scaled_coors >= [0, 0]) and np.all(scaled_coors < self.multilevel.out_size):
                coor = p
                break
        self.multilevel.cur_env.add_and_run(MovableStep(Point, coor, (), Point))
        return self.multilevel.cur_env.objs[-1]

    def get_tool_mask(self):
        return np.array(self.multilevel.tool_mask)[self.tool_list_reindex]



class InferenceConfig(gconfig.GeometryConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1






