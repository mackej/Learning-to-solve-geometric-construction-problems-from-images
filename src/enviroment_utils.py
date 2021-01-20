import cairo
import PIL
import os
from py_euclidea.tools import *


class EnvironmentUtils:

    @staticmethod
    def find_points(mask, bbox, number_of_points, manhattan_dist=20, threshold=0.5, opposite_corner_detection=True):
        """
        :param mask:
        :param number_of_points:
        :param manhattan_dist:
        :param threshold: threshold of minimum value to be significant point
        :param opposite_corner_detection if true then if number of points is 2 then one maximal is found and then
                2nd is symetry of bbox
        :return: return number of points that re most significant in mask,
        where every point have at least @manhattan_dist from each other.
        """
        mask = mask
        result = []
        while len(result) < number_of_points:
            pt_one = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
            if mask[pt_one] < threshold:
                break
            good_point = True
            for p in result:
                if abs(pt_one[0] - p[0]) + abs(pt_one[1] - p[1]) < manhattan_dist:
                    good_point = False
                    break
            if opposite_corner_detection and len(result) == 1 and number_of_points == 2:
                op_corner = EnvironmentUtils.get_other_corner(result[0], bbox)
                result.append(op_corner)
                good_point = False
                #if abs(pt_one[0] - op_corner[0]) + abs(pt_one[1] - op_corner[1]) > 3 * manhattan_dist:
                    #good_point = False

            if good_point:
                result.append(pt_one)
            else:
                mask[pt_one[0], pt_one[1]] = 0
        succession = False
        if len(result) == number_of_points:
            succession = True

        return np.array(result), succession

    @staticmethod
    def build_image(geom_primitives, out_size, scale, corners, goal_objects, history, visualisation=False):
        all_layer = EnvironmentUtils.objects_to_numpy(geom_primitives, out_size, scale, corners, visualisation=visualisation)
        goal_layer = EnvironmentUtils.objects_to_numpy(goal_objects, out_size, scale, corners, visualisation=visualisation)

        state = [all_layer, goal_layer]
        if len(history) == 0:
            state.append(np.zeros(out_size))
        for i in history:
            state.append(i.copy())
        return np.stack(state, axis=-1).astype(dtype=np.uint8)

    @staticmethod
    def build_image_from_multilevel_for_visualization(multi_level, history, scale, highlight_objects=[],
                            higlight_inputs=[]):
        highlight_ob_col = [0, 0, 204]
        highlight_inut_col = [204, 0, 204]
        visible = []
        for i in multi_level.cur_env.objs:
            if not hasattr(i, 'hidden') or i.hidden is False:
                visible.append(i)
        width, height = multi_level.out_size
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width * scale, height * scale)
        cr = cairo.Context(surface)
        cr.scale(*(scale / multi_level.scale))
        cr.set_source_rgb(1, 1, 1)
        cr.rectangle(0, 0, width * scale, height * scale)
        cr.fill()

        cr.set_source_rgb(0, 1, 0)
        for obj in multi_level.remaining_goals:
            obj.draw(cr, multi_level.corners, 1, True)

        cr.set_source_rgb(1, 0, 0)
        for obj in visible:
            obj.draw(cr, multi_level.corners, 1, True)

        for h_i in highlight_objects:
            cr.set_source_rgb(highlight_ob_col[0], highlight_ob_col[1], highlight_ob_col[2])
            h_i.draw(cr, multi_level.corners, 1, True)
        for h_i in higlight_inputs:
            cr.set_source_rgb(highlight_inut_col[0], highlight_inut_col[1], highlight_inut_col[2])
            h_i.draw(cr, multi_level.corners, 1, True)

        """SOLVED
                        Im unable to convert argb32 cairo to image to be able to show for debuging
                        and also for making output image. So i save cairo as image and then load it with
                        PIL from that file which is uneffective, but since this function runs only when exporting images
                        to presentation ill leave it like this.
        
                        surface.write_to_png("temporay_scene.png")
                        with PIL.Image.open("temporay_scene.png") as im:
                        image = np.array(im)
                        os.remove("temporay_scene.png")
                        return image.transpose(1, 0, 2)
                        + older version had RGB30 instead ARGB32
                """
        buf = surface.get_data()
        data = np.ndarray(shape=(width*scale, height*scale, 4), dtype=np.uint8, buffer=buf)

        return data[:, :, [2, 1, 0]].transpose(1, 0, 2)



    @staticmethod
    def objects_to_numpy(objs, out_size, scale, corners, visualisation=False):
        width, height = out_size
        surface = cairo.ImageSurface(cairo.FORMAT_A8, width, height)
        cr = cairo.Context(surface)
        cr.scale(*(1 / scale))

        cr.set_source_rgb(1, 1, 1)

        for obj in objs:
            #interection tool produce empty objects sometimes: TODO repair intersection objects
            if obj is not None:
                obj.draw(cr, corners, 1, visualisation)

        data = surface.get_data()
        data = np.array(data, dtype=np.uint8)
        data = data.reshape([height, surface.get_stride()])
        data = data[:, :width]
        return data.T

    @staticmethod
    def build_image_from_multilevel(multi_level, history):
        visible = []
        for i in multi_level.cur_env.objs:
            if not hasattr(i, 'hidden') or i.hidden is False:
                visible.append(i)
        return EnvironmentUtils.build_image(visible,
                                            multi_level.out_size,
                                            multi_level.scale,
                                            multi_level.corners,
                                            multi_level.remaining_goals,
                                            history
                                            )
    @staticmethod
    def dist(p1, p2):
        """retrun square of dist
        """
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    @staticmethod
    def get_midle_of_bbox(roi):
        return np.array([roi[0] + roi[2], roi[1] + roi[3]])/2

    @staticmethod
    def inside_roi(p, roi):
        return p[0] >= roi[0] and p[0] <= roi[2] and p[1] >= roi[1] and p[1] <= roi[3]

    @staticmethod
    def action_with_keypoint(prediction_index, tool_index, key_point_names, prediction, dataset):
        main_points, succession = EnvironmentUtils.find_points(prediction['masks'][:, :, prediction_index], prediction['rois'][prediction_index], len(key_point_names))

        if succession is False:
            return None, False

        roi = prediction['rois'][prediction_index]
        assigned_points = {}
        point_types = set(key_point_names)
        assigned = [False] * len(key_point_names)

        for key_point in point_types:
            possible_points = np.argwhere(prediction['class_ids'] == dataset.id_name_dic[key_point])
            if not key_point in assigned_points:
                assigned_points[key_point] = []

            for [p] in possible_points:
                point, succession = EnvironmentUtils.find_points(prediction['masks'][:, :, p], prediction['rois'][p], 1)
                if not succession:
                    continue
                point = point[0]
                if not EnvironmentUtils.inside_roi(point, roi):
                    continue

                dist_2 = np.sum((main_points - point) ** 2, axis=1)
                best_match = np.argmin(dist_2)
                if assigned[best_match] is False:
                    assigned_points[key_point].append(main_points[best_match])
                    assigned[best_match] = True

        main_points = [list(main_points[i]) for i in range(len(main_points))]
        result = [None] * len(key_point_names)
        for i in range(len(key_point_names)):
            result_class = key_point_names[i]
            if result_class in assigned_points and len(assigned_points[result_class])!=0:
                result[i] = assigned_points[result_class].pop(0)
                main_points.remove(list(result[i]))
        for i in range(len(key_point_names)):
            if result[i] is None:
                result[i] = main_points.pop(0)

        return np.array(result), True





    @staticmethod
    def action_from_prediction(prediction, dataset, environment_tools, action_type_hint=None,
                               use_oposite_corner_detection=True,
                               tool_mask=None, skip_those_indexes=[]):
        """
        :param prediction: prediction from network
        :param dataset:
        :param action_type_hint: hint which action should be done at this moment
        :param use_oposite_corner_detection if there is only one point for circle,
                use oposite corner in bbox of circle
        :param tool_mask: skip tools, that are no allowed in this tool mask
        :return:
        """

        not_effective_actions = ["Circle_CenterPoint", "Circle_RadiusPoint", "Angle_VertexPoint",
                                "Angle_RayPoint", "Perpendicular_ToLine", "Perpendicular_ThroughPoint",
                                "Parallel_ToLine", "Parallel_ThroughPoint", "CompassRadiusPoint", "CompassCenterPoint"]
        for i in range(len(prediction['class_ids'])):
            tool_index = prediction['class_ids'][i]
            name = dataset.class_names[tool_index]
            if i in skip_those_indexes:
                continue
            if name in not_effective_actions:
                continue
            env_tool_index = environment_tools[dataset.class_names[tool_index]]

            if tool_mask is not None and not tool_mask[env_tool_index]:
                continue
            if action_type_hint is not None and action_type_hint != env_tool_index:
                continue



            if name == 'Point':
                pts, succession = np.array(EnvironmentUtils.find_points(prediction['masks'][:, :, i], prediction['rois'][i], 1))
                if succession is False:
                    continue
                return env_tool_index, pts, i
            if name == 'Line' or name == "Perpendicular_Bisector":
                pts, succession = np.array(EnvironmentUtils.find_points(prediction['masks'][:, :, i], prediction['rois'][i], 2))
                if succession is False:
                    continue
                return env_tool_index, pts, i
            if name == 'Circle':
                reverse = False
                circle_points, succession = EnvironmentUtils.find_points(prediction['masks'][:, :, i], prediction['rois'][i], 2)

                if succession is False:
                    if len(circle_points) == 1 and use_oposite_corner_detection:
                        other_corner = np.array(circle_points[0]) + \
                                      2 * (EnvironmentUtils.get_midle_of_bbox(prediction['rois'][i]) - np.array(circle_points[0]))
                        circle_points = np.array([circle_points[0], other_corner])
                    else:
                        continue

                possible_points = np.argwhere(prediction['class_ids'] == dataset.id_name_dic['Circle_CenterPoint'])
                if len(possible_points) == 0:
                    reverse = True
                    possible_points = np.argwhere(prediction['class_ids'] == dataset.id_name_dic['Circle_RadiusPoint'])
                if len(possible_points) == 0:
                    return env_tool_index, circle_points, i
                best_dist = float("inf")
                best_dist_indexes = [0, 1]
                for [indc] in possible_points:
                    p, _ = EnvironmentUtils.find_points(prediction['masks'][:, :, indc], prediction['rois'][i], 1)
                    if len(p) == 0:
                        continue
                    [pt]=p
                    distances = [EnvironmentUtils.dist(pt, circle_points[0]),
                                 EnvironmentUtils.dist(pt, circle_points[1])]
                    min_index = distances.index(min(distances))
                    if distances[min_index] < best_dist:
                        best_dist = distances[min_index]
                        best_dist_indexes = [0, 1]
                        if min_index == 1:
                            best_dist_indexes = [1, 0]
                if reverse:
                    best_dist_indexes.reverse()
                return env_tool_index, np.array(circle_points)[best_dist_indexes], i
            if name == "Angle_Bisector":
                move, succession = EnvironmentUtils.action_with_keypoint(i, tool_index,
                        ["Angle_RayPoint", "Angle_VertexPoint", "Angle_RayPoint"], prediction, dataset)
                if succession is False:
                    continue
                return env_tool_index, move, i
            if name == "Perpendicular":
                move, succession= EnvironmentUtils.action_with_keypoint(i, tool_index,
                        ["Perpendicular_ToLine", "Perpendicular_ThroughPoint"], prediction, dataset)
                if succession is False:
                    continue
                return env_tool_index, move, i
            if name == "Parallel":
                move, succession= EnvironmentUtils.action_with_keypoint(i, tool_index,
                        ["Parallel_ToLine", "Parallel_ThroughPoint"], prediction, dataset)
                if succession is False:
                    continue
                return env_tool_index, move, i

            if name == "Compass":
                move, succession = EnvironmentUtils.action_with_keypoint(i, tool_index,
                        ["CompassRadiusPoint", "CompassRadiusPoint", "CompassCenterPoint"], prediction, dataset)
                if succession is False:
                    continue
                return env_tool_index, move, i

        return -1, [], -1

    @staticmethod
    def execute_one_step(pred, dataset, hint, multi_level):
        skips = []
        while True:
            action, points, pred_index = EnvironmentUtils.action_from_prediction(pred, dataset,
                                                                                 multi_level.tool_name_to_index,
                                                                                 action_type_hint=hint,
                                                                                 tool_mask=multi_level.tool_mask,
                                                                                 skip_those_indexes=skips)
            if action == -1:
                return None, None
            #print(action)
            r, done, tool_status, _ = EnvironmentUtils.complete_run_of_action(action, points, multi_level)
            if tool_status is True:
                break
            else:
                skips.append(pred_index)
        return r, done
    @staticmethod
    def prepare_all_hypothesis(pred, dataset, multi_level):
        hypothesis = []
        skips = []
        while True:
            action, points, pred_index = EnvironmentUtils.action_from_prediction(pred, dataset,
                                                                                 multi_level.tool_name_to_index,
                                                                                 tool_mask=multi_level.tool_mask,
                                                                                 skip_those_indexes=skips)
            if action == -1:
                return hypothesis
            skips.append(pred_index)
            hypothesis.append(EnvironmentUtils.hypothesis_to_primitive_interactions(action, points, multi_level,
                                                                                    pred["scores"][pred_index]))

    @staticmethod
    def hypothesis_to_primitive_interactions(action, points, multi_level, score):
        r, done, tool_status, action_size = EnvironmentUtils.complete_run_of_action(action, points, multi_level,
                                                                                    change_goals=False)
        env = multi_level.cur_env
        start = len(env.steps) - action_size
        created_steps = env.steps[start:]
        result_objs = []
        for i in created_steps:
            for j in i.output:
                result_objs.append(env.objs[j])

        EnvironmentUtils.reverte_last_action(action_size, multi_level)
        return {"reward": r, "done": done, "action_id": action, "tool_status": tool_status, "steps": created_steps,
                "result_obj": result_objs, "score":score}

    @staticmethod
    # This method will create all points that are necessary to execute main action.
    def complete_run_of_action(action, points, multi_level, change_goals=True):
        r_complete = 0
        number_of_suc_actions = 0
        # 1 + len(points) is maximum possible number of subactions: 1 main and possible point tool action for each of
        # click points
        for i in range(1 + len(points)):
            multi_level.action_set_tool(action)
            for pt in range(points.shape[0]):
                r, done, tool_status = multi_level.action_click_point(points[pt], change_rem_goals=change_goals)
                if tool_status is False and multi_level.tool_mask[0]:
                    multi_level.action_set_tool(0)
                    r, done, tool_status = multi_level.action_click_point(points[pt], change_rem_goals=change_goals)
                    if tool_status is True:
                        number_of_suc_actions += 1
                        r_complete += r
                    break
                if tool_status is True:
                    return r + r_complete, done, True, number_of_suc_actions+1
        EnvironmentUtils.reverte_last_action(number_of_suc_actions, multi_level)
        return 0, False, False, 0

    @staticmethod
    def reverte_last_action(action_size, multi_level):
        for _ in range(action_size):
            multi_level.cur_env.pop()





    @staticmethod
    def get_other_corner(point, roi):
        other_corner = np.array(point) + \
                       2 * (EnvironmentUtils.get_midle_of_bbox(roi) - np.array(point))
        return other_corner

    @staticmethod
    def compare_hypothesis(h1, h2):
        if h1['reward'] != h2['reward'] or h1["done"] != h2["done"]:
            return False
        if len(h1['result_obj']) != len(h2['result_obj']) or len(h2['steps']) != len(h1['steps']):
            return False
        # for lower number of hypothesis we compare only final resulting object
        '''
        for i in range(len(h1['result_obj'])):
            if not h1['result_obj'][i].identical_to(h2['result_obj'][i]):
                return False
        '''
        if not h1['result_obj'][-1].identical_to(h2['result_obj'][-1]):
            return False
        '''
        for i in range(len(h1['steps'])):
            if type(h1['steps'][i]) != type(h2['steps'][i]):
                return False
            if h1['steps'][i].args_i != h2['steps'][i].args_i:
                return False
            if h1['steps'][i].otypes != h2['steps'][i].otypes:
                return False
        '''
        return True

