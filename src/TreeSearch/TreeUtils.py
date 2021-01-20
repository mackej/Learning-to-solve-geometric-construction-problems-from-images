from GeometryObjectsDetector.DetectorUtils import *
from py_euclidea.tools import *
from py_euclidea.environment import ConstrStep, MovableStep
from enviroment_utils import  *
from mrcnn import visualize
from TreeSearch import State
import matplotlib.pyplot as plt
import itertools
from PIL import Image
import numpy as np

from TreeSearch.tools_degrees_of_freedom import *
from TreeSearch.Action import *


def add_one_edge(ordered_objs, res, multilevel, a, tool):
    valid_action = True
    for i in range(len(ordered_objs)):
        ob = ordered_objs[i]
        t = tool.in_types[i]
        #if ob is None or np.any(np.isnan(ob.data)):
            #return

        if t == Point and not isinstance(ob, Point):
            point_coor = reasonable_random_point_on(multilevel, ob)
            # we create 1 more point so the action need one more object
            a.additional_point_actions.append(point_coor)
            a.in_objects.append(None)
            a.action_obj_cnt += 1
        else:
            if isinstance(ob, t):
                a.in_objects.append(ob.index)
            else:
                valid_action = False
                break
    if valid_action:
        res.append(a)

def get_all_possible_actions(all_objects, multilevel, dataset, possible_tools, freedom_degrees):
    res = []

    # adding n tuples without repeats
    for tool, tool_name in possible_tools:
        for objs in itertools.combinations(all_objects, len(tool.in_types)):
            degrees = freedom_degrees[tool_name]
            for d in degrees:
                a = Action(tool, tool_name)
                ordered_objs = np.array(objs)[d]
                add_one_edge(ordered_objs, res, multilevel, a, tool)
    # adding 1-tuples with repeats equal to number of arguments (2,3)
    for tool, tool_name in possible_tools:
        for obj in all_objects:
            a = Action(tool, tool_name)
            ordered_objs = [obj] * len(tool.in_types)
            add_one_edge(ordered_objs, res, multilevel, a, tool)
    # if we have tool with 3 argmunets: one allowe repeat. ....we can generate all 2-tuples and
    # then make all 4 combinations how to add one repeat
    repeat_combinations = [[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1]]
    for tool, tool_name in possible_tools:
        if len(tool.in_types) != 3:
            continue
        for objs in itertools.combinations(all_objects, 2):
            for d in repeat_combinations:
                a = Action(tool, tool_name)
                ordered_objs = np.array(objs)[d]
                add_one_edge(ordered_objs, res, multilevel, a, tool)

    return res


def reasonable_random_point_on(multilevel, ob):
    # TODO: improve random object genertation
    while True:
        if isinstance(ob, Circle):
            p = ob.r * unit_vector(np.random.random() * 2 * np.pi) + ob.c
        if isinstance(ob, Line):
            offset = 0.1
            coef = np.random.uniform(offset, 1 - offset)
            x, y = ob.get_endpoints(multilevel.corners)
            p = coef * x + (1 - coef) * y

        scaled_coors = p / multilevel.scale
        if np.all(scaled_coors >= [0, 0]) and np.all(scaled_coors < multilevel.out_size):
            coor = p
            break
    #multilevel.cur_env.add_and_run(MovableStep(Point, coor, (), Point))
    return coor

def execute_action(multilevel, action, last_image, detected_objects, change_rem_goals=True,):
    arguments = []
    nr = 0
    for i in action.in_objects:
        if i is not None:
            arguments.append(i)
        else:
            multilevel.cur_env.add_and_run(MovableStep(Point, action.additional_point_actions[nr], (), Point))
            # detect new point since we know click coors so we dont have to redetect point later
            _, new_point = multilevel.cur_env.closest_obj_of_type(action.additional_point_actions[nr], Point, include_goal=False)
            detected_objects.append(new_point)
            nr+=1
            arguments.append(multilevel.cur_env.objs[-1].index)

    tool_status = action.tool.run_known_objects(multilevel.cur_env, arguments)
    if tool_status is False:
        return -1, False, False, detected_objects
    #print(tool_status)
    new_image = EnvironmentUtils.build_image_from_multilevel(multilevel, [])
    suc, new_objects = objects_based_on_img_diff(last_image, new_image, action.tool, multilevel)
    for i in new_objects:
        detected_objects.append(i)
    if suc is False:
        return -1, False, False, detected_objects
    r, done = multilevel.evaluate_last_step(change_rem_goals=change_rem_goals)
    return r, done, True, detected_objects

def reverse_to_state(prev_state, cur_state, all_objects, multilevel):
    for _ in range(cur_state.env_objects_len - prev_state.env_objects_len):
        multilevel.cur_env.pop()
    for _ in range(cur_state.all_objects_len - prev_state.all_objects_len):
        all_objects.pop()




def objects_based_on_img_diff(prev_image, next_image, tool, multilevel, args):
    '''
    Based on differnce between last 2 state images find new objects in scene, also return false if
    there is no diffrence which means we didn't draw anything new. just draw same geom primituve with different tool
    or based on different objects
    :param prev_image:
    :param next_image:
    :param tool:
    :param multilevel:
    :param args args used to run tool
    :return:
    '''
    diff = np.array(next_image[:, :, 0] > 0, dtype=int) - np.array(prev_image[:, :, 0] > 0, dtype=int)
    #diff = next_image[:, :, 0] -  prev_image[:, :, 0]
    m = np.array(diff) > 0
    if not np.any(m):
        return False, []
    #im = np.zeros((256,256,3),dtype=next_image.dtype)
    #im[:,:,0] = m*120
    #Image.fromarray(im, 'RGB').show()
    out_t = tool.get_out_types(multilevel.cur_env, args)
    if not isinstance(out_t, list):
        out_t = [out_t]
    if len(out_t) > 1:
        # Should be only intersection tool and result should be [point, point]
        p1 = DetectorUtils.match(m, out_t[0], multilevel, include_goal=False)
        p1_c = [int(x) for x in p1.a / multilevel.scale]
        del_size = 5
        m[p1_c[0]-del_size:p1_c[0]+del_size, p1_c[1]-del_size:p1_c[1]+del_size] = np.zeros((del_size*2, del_size*2))
        if not np.any(m):
            return True, [p1]
        p2 = DetectorUtils.match(m, out_t[1], multilevel, include_goal=False)
        return True, [p1, p2]
    else:
        return True, [DetectorUtils.match(m, out_t[0], multilevel, include_goal=False)]



def get_all_moves(multilevel, detector_model, dataset, freedom_degrees, usable_tools, detected_images = None):
    '''
    :param multilevel:
    :param detector_model:
    :param dataset:
    :param freedom_degrees:
    :param usable_tools:
    :param detected_images: when None this method will use model to detect objects in scene. If thi is an array with
        detected objects then we can skip detection and use those
    :return:
    '''
    image = EnvironmentUtils.build_image_from_multilevel(multilevel, [])
    image[:, :, 1] = 0
    if detected_images is None:
        detection = detector_model.detect([image], verbose=0, bool_masks=False)[0]
        construction_indexes = detection["class_ids"] <= 3
        detection["masks"] = detection["masks"][:, :, construction_indexes]
        detection["class_ids"] = detection["class_ids"][construction_indexes]
        all_objects, acc = DetectorUtils.find_environment_objects(detection, multilevel, dataset, include_goal=False)
        visualize.display_instances(image, detection['rois'], detection['masks'] > 0.5, detection['class_ids'],
                                dataset.class_names, detection['scores'],
                                show_mask=True, show_bbox=True, ax=get_ax(), )
    else:
        all_objects = detected_images

    actions = get_all_possible_actions(all_objects, multilevel, dataset, usable_tools, freedom_degrees)
    initial_number_of_detetcted_objects = len(all_objects)
    initial_number_of_objects_in_env = len(multilevel.cur_env.objs)
    correct_actions = []
    for a in actions:
        r, done, succes, all_objects = execute_action(multilevel, a, image, all_objects, change_rem_goals=False)
        reverse_n_steps_env(len(multilevel.cur_env.objs) - initial_number_of_objects_in_env, multilevel)
        reverse_n_steps_actions(len(all_objects) - initial_number_of_detetcted_objects, all_objects)
        a.action_reward = r
        # execute of the action makes no difference inside environment or some internal error has occurred.
        if succes:
            correct_actions.append(a)

    actions = sorted(correct_actions, key=lambda x: x.action_reward, reverse=True)
    return actions

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(800/90, 800/90))
    return ax


