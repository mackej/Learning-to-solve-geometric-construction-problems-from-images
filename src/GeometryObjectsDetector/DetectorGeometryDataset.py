import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from enviroment_utils import EnvironmentUtils as env_utils
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from py_euclidea import geo_object
from mrcnn import utils


class DetectorGeometry(utils.Dataset):
    """Image generator. We store just geometric primitives, that define
    each state and then we each time build image just from those primitives
    """
    def get_number_od_classes(self):
        return len(self.id_name_dic)

    def PrepareDataGen(self, mask_size=5 ,history_size=0,):
        self.mask_size = mask_size
        self.history_size = history_size
        self.last_id = 0
        self.problem_name = "DetectorGeometryDataet"

        self.add_class(self.problem_name, 1, "Construction_Point")
        self.add_class(self.problem_name, 2, "Construction_Line")
        self.add_class(self.problem_name, 3, "Construction_Circle")
        self.add_class(self.problem_name, 4, "Goal_Point")
        self.add_class(self.problem_name, 5, "Goal_Line")
        self.add_class(self.problem_name, 6, "Goal_Circle")
        self.number_of_non_history_classes = 6
        for i in range(self.history_size):
            self.add_class(self.problem_name, 6+i+1, "LastMove_"+str(i+1))

        self.id_name_dic = {}
        for info in self.class_info:
            self.id_name_dic[info['name']] = info['id']

        #self.add_class("shapes", 3, "triangle")

    def add_one_scene(self, geom_primitives, multi_level, environment_run_id):
        remaining_goals = []
        for g in multi_level.remaining_goals:
            remaining_goals.append(g.copy())
        self.add_image(self.problem_name, image_id=self.last_id, path=None,
                       geom_primitives=geom_primitives, goal_objects=remaining_goals,
                       out_size=multi_level.out_size, scale=multi_level.scale, corners=multi_level.corners,
                       environment_run_id=environment_run_id)
        self.last_id += 1

    def load_image(self, image_id):
        """Build np array from primitives and goals
        """
        info = self.image_info[image_id]
        geom_primitives = info['geom_primitives']
        run_id = info['environment_run_id']
        goal_objects = info['goal_objects']
        out_size = info['out_size']
        scale = info['scale']
        corners = info['corners']
        history = self.get_history(image_id)

        return env_utils.build_image(geom_primitives, out_size, scale, corners, goal_objects, history)

    def get_history(self, image_id):
        info = self.image_info[image_id]
        run_id = info['environment_run_id']
        out_size = info['out_size']

        history = []
        history_id = image_id-1
        while run_id == self.image_info[history_id]['environment_run_id'] and\
                len(history) < self.history_size\
                and history_id >= 0:
            hist_info = self.image_info[history_id]
            hist_img = env_utils.objects_to_numpy(hist_info['geom_primitives'],
                                                        hist_info['out_size'],
                                                        hist_info['scale'],
                                                        hist_info['corners'])
            history.append(hist_img)
        for i in range(self.history_size - len(history)):
            history.append(np.zeros(out_size))
        history.reverse()
        return history

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == self.problem_name:
           return image_id
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        heat_map = True when to use heat map false to square mask
        """
        info = self.image_info[image_id]
        geom_primitives = info['geom_primitives']
        run_id = info['environment_run_id']
        goal_objects = info['goal_objects']
        out_size = info['out_size']
        scale = info['scale']
        corners = info['corners']
        mask = []
        ids = []
        for i in geom_primitives:
            if hasattr(i, 'hidden') and i.hidden is True:
                continue
            m = i.get_mask(corners, out_size, scale, self.mask_size)
            mask.append(m)
            id = self.id_name_dic[i.get_name()]
            ids.append(id)
        for i in goal_objects:
            if hasattr(i, 'hidden') and i.hidden is True:
                continue
            m = i.get_mask(corners, out_size, scale, self.mask_size)
            mask.append(m)
            id = self.id_name_dic["Goal_"+i.get_name()]
            ids.append(id)
        for i in range(self.history_size):
            if image_id-i-1 < 0 or self.image_info[image_id-i-1]['environment_run_id'] != run_id:
                break
            m = geom_primitives[len(geom_primitives)-i-1].get_mask(corners, out_size, scale, self.mask_size)
            id = self.number_of_non_history_classes + 1 + i
            mask.append(m)
            ids.append(id)

        return np.stack(mask, axis=-1), np.array(ids, dtype=np.int32)



