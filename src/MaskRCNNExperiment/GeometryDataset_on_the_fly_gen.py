import os
import sys
import numpy as np
from collections import deque
import pickle
from py_euclidea import multi_level
from LevelSelector import LevelSelector
from skimage import draw
from scipy.stats import multivariate_normal
import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

from enviroment_utils import EnvironmentUtils as env_utils
from MaskRCNNExperiment import GeometryDataset
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from py_euclidea import geo_object
from mrcnn import utils


class GeometryDataset_on_the_fly_gen(GeometryDataset.GeometryDataset):

    def __init__(self,  args):
        super(GeometryDataset.GeometryDataset, self).__init__()
        levels = LevelSelector.get_levels(
            match=args.generate_levels
        )
        m = multi_level.MultiLevel((levels), number_of_construction_sub_goals=args.number_of_partial_goals)
        min_tool_set = None
        if args.tool_set == "min_by_levels":
            min_tool_set = m.get_min_set_of_tool()
        if args.tool_set == "min_by_construction":
            min_tool_set = m.get_min_set_of_tools_for_constructions()
        self.multi_level = m
        self.PrepareDataGen(use_heat_map=args.use_heat_map > 0, heat_map_size=args.mask_size,
                                history_size=args.history_size, heat_map_cov=args.heat_map_covariance,
                                tool_list=min_tool_set)
        self.add_image(self.problem_name, image_id=self.last_id, path=None,
                       action_points=[], tool_id=0)
        print("resetting on-the-fly statistics.")
        self.number_of_images = 0
        self.number_of_levels = 0
        self.number_of_errors = 0
        self.prepare()


    def prepare(self, class_map=None):
        self.multi_level.next_level()
        self.step_index = -1
        self.history = deque(maxlen=self.history_size)
        self.done_level = False
        super().prepare(class_map=class_map)

    def reset_history(self):
        for i in range(self.history_size):
            self.history.append(np.zeros(self.multi_level.out_size))

    def execute_action(self, action_tool, action_points):
        self.multi_level.action_set_tool(action_tool)
        try:
            for pt in action_points:
                r, done, tool_status = self.multi_level.action_click_point(pt + 0.1, auto_proceed=False)
                if tool_status is False and self.multi_level.tool_mask[0]:
                    self.multi_level.action_set_tool(0)
                    r, done, tool_status = self.multi_level.action_click_point(pt)
                    break
            if done:
               self.done_level = True
        except Exception as e:
            print("error in inferrence from model data.")
            with open("err_in_training_{}.txt".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")), mode="a+",encoding='utf-8') as f:
                print("error in inferrence from model data.")
                print(e)
            # perform file operations
            #raise e

    def reset_level(self):
        self.done_level = False
        self.multi_level.next_level()
        self.step_index = 0
        self.reset_history()
        self.number_of_levels += 1

    def load_image(self, image_id):
        self.number_of_images += 1
        self.step_index += 1
        if self.done_level:
            self.reset_level()

        action_tool, action_points = self.multi_level.get_construction(self.step_index)

        if action_tool is None and action_points is None:
            print("error in generation")
            self.reset_level()
            self.number_of_errors +=1
            action_tool, action_points = self.multi_level.get_construction(self.step_index)

        action_tool_network_index = self.id_name_dic[self.multi_level.tool_index_to_name[action_tool]]

        result = env_utils.build_image_from_multilevel(self.multi_level, self.history)
        self.history.append(result[:, :, 0])
        self.execute_action(action_tool, action_points)

        # alter info so the mas generation is ame like in parent class
        self.image_info[image_id]['tool_id'] = action_tool_network_index
        self.image_info[image_id]['action_points'] = action_points

        return result


    def image_reference(self, image_id):
        print("this method should be called, since data are generated 'on-the-fly'")
        info = self.image_info[image_id]
        if info["source"] == self.problem_name:
           return image_id
        else:
            super(self.__class__).image_reference(self, image_id)

    def get_statistics(self):
        return "{} images were generated. Total of {} different levels. {} errors in training data".format(
            self.number_of_images, self.number_of_levels, self.number_of_errors)





