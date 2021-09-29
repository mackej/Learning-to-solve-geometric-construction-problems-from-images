import os
import sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

from enviroment_utils import EnvironmentUtils as env_utils
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from py_euclidea import geo_object
from mrcnn import utils


class GeometryDataset(utils.Dataset):
    """Image generator. We store just geometric primitives, that define
    each state and then we each time build image just from those primitives
    """
    dump_class_info_file = "class_info_dump.pckl"
    def dump_class_info(self, dir):
        pickle.dump(self.class_info, open(os.path.join(dir, self.dump_class_info_file), "wb"))


    def get_number_od_classes(self):
        return len(self.id_name_dic)

    def PrepareDataGen(self, use_heat_map=True, heat_map_size=40, history_size=0, heat_map_cov=100, tool_list=None, tool_list_file=None):
        self.heat_map_cov = heat_map_cov
        self.history_size = history_size
        self.use_heat_map = use_heat_map
        self.heat_map_size = heat_map_size
        self.heat_map = None
        self.last_id = 0
        self.problem_name = "GeometryFromImages"

        all_types = {"Line": 1,
                     "Circle": 2,
                     "Perpendicular_Bisector": 3,
                     "Angle_Bisector": 4,
                     "Perpendicular": 5,
                     "Parallel": 6,
                     "Compass": 7,
                     "Circle_CenterPoint": 8,
                     "Circle_RadiusPoint": 9,
                     "Angle_VertexPoint": 10,
                     "Angle_RayPoint": 11,
                     "Perpendicular_ToLine": 12,
                     "Perpendicular_ThroughPoint": 13,
                     "Parallel_ToLine": 14,
                     "Parallel_ThroughPoint": 15,
                     "CompassRadiusPoint": 16,
                     "CompassCenterPoint": 17,
                     "Point": 18}
        connected_tools = {"Circle": ["Circle_CenterPoint", "Circle_RadiusPoint"],
                           "Angle_Bisector": ["Angle_VertexPoint", "Angle_RayPoint"],
                           "Perpendicular": ["Perpendicular_ToLine", "Perpendicular_ThroughPoint"],
                           "Parallel": ["Parallel_ToLine", "Parallel_ThroughPoint"],
                           "Compass": ["CompassRadiusPoint", "CompassCenterPoint"]
                           }
        possible_dump_info = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.dump_class_info_file)
        if tool_list_file is not None:
            possible_dump_info = os.path.join(os.path.realpath(tool_list_file), self.dump_class_info_file)
        if os.path.exists(possible_dump_info):
            self.class_info = pickle.load(open(possible_dump_info, "rb"))
        else:
            if tool_list is not None:
                tool_set = {}
                for i in tool_list:
                    tool_set[i] = True
                    if i in connected_tools:
                        for c in connected_tools[i]:
                            tool_set[c] = True
                tool_set_k = sorted(tool_set.keys(), key=lambda k: all_types[k])
                for k in tool_set_k:
                    self.add_class(self.problem_name, len(self.class_info), k)
            else:
                for i in all_types.keys():
                    self.add_class(self.problem_name, all_types[i], i)

        self.id_name_dic = {}
        #print(self.class_info, flush=True)
        for info in self.class_info:
            self.id_name_dic[info['name']] = info['id']

        #self.add_class("shapes", 3, "triangle")

    def add_one_scene(self, geom_primitives, action_points, tool_id, multi_level, environment_run_id):
        remaining_goals = []
        for g in multi_level.remaining_goals:
            remaining_goals.append(g.copy())
        self.add_image(self.problem_name, image_id=self.last_id, path=None,
                       geom_primitives=geom_primitives, goal_objects=remaining_goals,
                       action_points=action_points, tool_id=tool_id,
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
        action_points = info['action_points']
        tool_id = info['tool_id']
        return self.procces_mask(action_points, tool_id)

    def procces_mask(self, action_points, tool_id):
        #point is same as must but just different id
        if ('Line' in self.id_name_dic and self.id_name_dic['Line'] == tool_id) or\
            ('Point' in self.id_name_dic and self.id_name_dic['Point'] == tool_id) or\
            ('Perpendicular_Bisector' in self.id_name_dic and self.id_name_dic['Perpendicular_Bisector'] == tool_id):
            return self.line_mask(action_points, tool_id)

        if 'Circle' in self.id_name_dic and self.id_name_dic['Circle'] == tool_id:
            return self.circle_mask(action_points, tool_id)

        if 'Angle_Bisector' in self.id_name_dic and self.id_name_dic['Angle_Bisector'] == tool_id:
            return self.mask_with_multiple_objects(action_points, tool_id,
                                                   ["Angle_RayPoint", "Angle_VertexPoint", "Angle_RayPoint"])

        if 'Perpendicular' in self.id_name_dic and self.id_name_dic['Perpendicular'] == tool_id:
            return self.mask_with_multiple_objects(action_points, tool_id,
                                                   ["Perpendicular_ToLine",  "Perpendicular_ThroughPoint"])

        if 'Parallel' in self.id_name_dic and self.id_name_dic['Parallel'] == tool_id:
            return self.mask_with_multiple_objects(action_points, tool_id,
                                                   ["Parallel_ToLine",  "Parallel_ThroughPoint"])

        if 'Compass' in self.id_name_dic and self.id_name_dic['Compass'] == tool_id:
            return self.mask_with_multiple_objects(action_points, tool_id,
                                                   ["CompassRadiusPoint", "CompassRadiusPoint", "CompassCenterPoint"])

    def circle_mask(self, action_points, tool_id):
        target_mask = np.zeros((256, 256, 3), dtype=np.float32)
        for p in action_points:
            self.add_point_mask(target_mask, 0, p)

        self.add_point_mask(target_mask, 1, action_points[0])
        self.add_point_mask(target_mask, 2, action_points[1])

        return target_mask, np.array(
            [tool_id,
             self.id_name_dic['Circle_CenterPoint'],
             self.id_name_dic['Circle_RadiusPoint']], dtype=np.int32)

    def mask_with_multiple_objects(self, action_points, tool_id, list_of_detections):
        '''
        :param action_points:
        :param tool_id:
        :param list_of_detections: list of detections like ["Parallel_ToLine","Parallel_ThroughPoint"] for
            paraller tol for example
        :return:
        '''
        target_mask = np.zeros((256, 256, 1+len(list_of_detections)), dtype=np.float32)
        for p in action_points:
            self.add_point_mask(target_mask, 0, p)
        target_labels = [tool_id]
        for i in range(1,1+len(list_of_detections)):
            self.add_point_mask(target_mask, i, action_points[i-1])
            target_labels.append(self.id_name_dic[list_of_detections[i-1]])
        return target_mask, np.array(target_labels, dtype=np.int32)


    def line_mask(self, action_points, tool_id):
        target_mask = np.zeros((256, 256, 1), dtype=np.float32)
        for p in action_points:
            self.add_point_mask(target_mask, 0, p)
        return target_mask, np.array([tool_id], dtype=np.int32)

    @staticmethod
    def build_heat_mat_point(heat_map_size, cov=100):
        k = multivariate_normal(mean=0, cov=cov)

        v = np.zeros((2*heat_map_size+1, 2*heat_map_size+1))
        for x in range(heat_map_size*2+1):
            for y in range(heat_map_size*2+1):
                dist = np.sqrt((x-heat_map_size+1) ** 2 + (y-heat_map_size+1) ** 2)
                v[x, y] = k.cdf(-dist) * 2
        return v

    def add_point_mask(self, mask, index, point, heat_map=True):
        """
        modify mask at @index adding point
        :param mask: [x, y, channel]
        :param index: channel index of mask where point should be made
        :param point: [X, Y] [float, int] coordinates of point
        :return:
        """
        if self.use_heat_map and self.heat_map is None:
            self.heat_map = GeometryDataset.build_heat_mat_point(self.heat_map_size, self.heat_map_cov)
        if self.use_heat_map:
            submask = self.heat_map
        else:
            submask = np.ones((2*self.heat_map_size+1, 2*self.heat_map_size+1))


        px_1 = int(point[0]) - self.heat_map_size
        py_1 = int(point[1]) - self.heat_map_size
        px_2 = int(point[0]) + self.heat_map_size + 1
        py_2 = int(point[1]) + self.heat_map_size + 1

        px_1_shift = max(0, -px_1)
        py_1_shift = max(0, -py_1)
        px_2_shift = 0 if px_2 < mask.shape[0] else mask.shape[0] - px_2
        py_2_shift = 0 if py_2 < mask.shape[1] else mask.shape[1] - py_2

        mask[px_1+px_1_shift:px_2+px_2_shift, py_1+py_1_shift:py_2+py_2_shift, index]\
            = np.max(np.array([submask[
                               px_1_shift:submask.shape[0]+px_2_shift,
                               py_1_shift:submask.shape[1]+py_2_shift
                               ],
                               mask[px_1+px_1_shift:px_2+px_2_shift, py_1+py_1_shift:py_2+py_2_shift, index]]), axis=0)



