import numpy as np
from py_euclidea.constructions import *

class DetectorUtils:
    @staticmethod
    def find_environment_objects(prediction, multi_level, dataset, number_of_samples_fr_each_object=10,include_goal=True):

        matching = []
        for i in range(len(prediction['class_ids'])):
            mask = prediction['masks'][:, :, i]
            class_id = prediction['class_ids'][i]
            matching.append(DetectorUtils.match_prediction(mask, class_id, dataset, multi_level,
                                                           number_of_samples_fr_each_object=
                                                           number_of_samples_fr_each_object,
                                                           include_goal=include_goal))
        number_of_founded_objects = 0
        number_of_visible_objects = 0
        for ob in multi_level.cur_env.objs:
            if hasattr(ob, "hidden") and ob.hidden:
                continue
            number_of_visible_objects += 1
            if ob in matching:
                number_of_founded_objects += 1
        # todo constructed  objects are counted in goal.... remain in cur_goal()
        for ob in multi_level.remaining_goals:
            if hasattr(ob, "hidden") and ob.hidden:
                continue
            number_of_visible_objects += 1
            if ob in matching:
                number_of_founded_objects += 1
        return matching, number_of_founded_objects / number_of_visible_objects

    @staticmethod
    def match_prediction(mask, class_id, dataset, multi_level, number_of_samples_fr_each_object=10, include_goal=True):
        objects = {"Point": Point, "Line": Line, "Circle": Circle}
        class_name = dataset.class_names[class_id]
        adj, object_type = class_name.split("_")
        return DetectorUtils.match(mask, objects[object_type], multi_level, number_of_samples_fr_each_object=10, include_goal=True)

    @staticmethod
    def match(mask, ob_type,multi_level, number_of_samples_fr_each_object=10, include_goal=True):
        ix, iy = np.where(mask > 0.5)
        p = np.random.permutation(len(ix))
        rnd_indexes = p[:number_of_samples_fr_each_object]
        match = {}
        for index in rnd_indexes:
            coor = np.array([ix[index], iy[index]]) * multi_level.scale
            dist, obj = multi_level.cur_env.closest_obj_of_type(coor, ob_type, include_goal=include_goal)
            if dist > 50:
                continue
            if obj not in match:
                match[obj] = 0
            match[obj] += 1
        match_with = list(match.keys())[np.argmax(match.values())]
        return match_with
