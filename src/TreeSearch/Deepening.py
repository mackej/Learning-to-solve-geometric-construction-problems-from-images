
from TreeSearch.TreeUtils import *
from TreeSearch.tools_degrees_of_freedom import *

class Deepening:
    def __init__(self, starting_depth=5, max_depth=10):
        self.starting_depth = starting_depth
        self.max_depth = max_depth
        self.depth = 0
        self.cur_max_depth = 1
        self.freedom_degrees = get_degrees_of_freedom()

    def search(self, multilevel, dataset, detector_model, all_objects, depth, path):
        image = EnvironmentUtils.build_image_from_multilevel(multilevel, [])
        Image.fromarray(image, 'RGB').show()
        if self.cur_max_depth < depth:
            return [], False
        moves = get_all_moves(multilevel, detector_model, dataset, self.freedom_degrees, self.usable_tools, detected_images=all_objects)
        for m in moves:
            if self.action_already_used(m, path):
                # do not use same action over again
                continue
            r, done, suc, all_objects = execute_action(multilevel, m, image, all_objects)
            path.append(m)
            if done:
                reverse_last_action(multilevel, m)
                return path, True
            res, sol = self.search(multilevel, dataset, detector_model,all_objects, depth+1, path)
            if sol:
                reverse_last_action(multilevel, m)
                return res, True
            reverse_last_action(multilevel, m)
            path.pop()
        return [], False
    @staticmethod
    def action_already_used(action, past_actions):
        for ac in past_actions:
            if action.tool_name == ac.tool_name and action.in_objects == ac.in_objects:
                return True
        return False

    def initialize_search(self, multilevel, detector_model, dataset):
        image = EnvironmentUtils.build_image_from_multilevel(multilevel, [])
        # Image.fromarray(image, 'RGB').show()
        image[:, :, 1] = 0
        detection = detector_model.detect([image], verbose=0, bool_masks=False)[0]
        visualize.display_instances(image, detection['rois'], detection['masks'] > 0.5, detection['class_ids'],
                                    dataset.class_names, detection['scores'],
                                    show_mask=True, show_bbox=True, ax=get_ax(), )
        tools = []
        for tool_name in multilevel.cur_env.enabled_tools:
            if tool_name == "move" or tool_name == "Point":
                continue
            tool_index = multilevel.tool_name_to_index[tool_name]
            tools.append([multilevel.tools[tool_index], tool_name])
        self.usable_tools = tools
        all_objects, acc = DetectorUtils.find_environment_objects(detection, multilevel, dataset, include_goal=False)
        return all_objects

    def deepening(self, multilevel, dataset, detector_model):
        all_objects = self.initialize_search(multilevel, detector_model, dataset)
        for i in range(self.starting_depth, self.max_depth):
            self.cur_max_depth = i
            path, sol = self.search(multilevel, dataset, detector_model, all_objects, 0, [])
            if sol:
                return path
        return []
