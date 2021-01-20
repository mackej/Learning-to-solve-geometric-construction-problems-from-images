import sys
from TreeSearch.TreeUtils import *
from TreeSearch.tools_degrees_of_freedom import *
from TreeSearch.State import *
import TreeSearch.enumerate_estimate_of_branching_factor

class IterativeDeepening:
    def __init__(self,  detect_features=True, tool_hints=False):
        '''
        :param starting_depth:
        :param max_depth:
        :param detect_features: whether we ave to detect geom primitives. False when we get them str8 from environment
        '''
        # max_node_visits: if we acces more odes then search returns fail... as if solution does not exist
        self.max_node_visits = 10_000
        # currently visited nodes
        self.cur_visits = 0
        self.detect_features = False
        self.cur_max_depth = 1
        self.freedom_degrees = get_degrees_of_freedom()
        self.multilevel = None
        self.dataset = None
        self.all_objects = None
        self.detector_model = None
        self.usable_tools = None
        self.use_hint = tool_hints
        self.branch_factor_sum = {}
        self.branch_level_visits = {}
        self.geom_primitives = {}
        self.goal_size = 0


    @staticmethod
    def reverse_to_state(prev_state, all_objects, multilevel):
        #print(prev_state.all_objects_len, " ", prev_state.env_objects_len)
        cur_state = State(multilevel, all_objects, build_image=False)
        while len(multilevel.cur_env.objs) > prev_state.env_objects_len:
            multilevel.cur_env.pop()
        for _ in range(cur_state.all_objects_len - prev_state.all_objects_len):
            all_objects.pop()

    @staticmethod
    def action_already_used(action, past_actions):
        for ac in past_actions:
            if action.tool_name == ac.tool_name and action.in_objects == ac.in_objects:
                return True
        return False

    def execute_action(self, action, last_image, change_rem_goals=True, ):
        #print("pre execute", self.multilevel.cur_env.objs)
        arguments = []
        nr = 0
        for i in action.in_objects:
            if i is not None:
                arguments.append(i)
            else:
                self.multilevel.cur_env.add_and_run(MovableStep(Point, action.additional_point_actions[nr], (), Point))
                # detect new point since we know click coors so we dont have to redetect point later
                _, new_point = self.multilevel.cur_env.closest_obj_of_type(action.additional_point_actions[nr], Point,
                                                                      include_goal=False)

                self.all_objects.append(new_point)
                nr += 1
                arguments.append(self.multilevel.cur_env.objs[-1].index)

        # if we use point tool while creating this action re-render object for difference detection.
        # Which also cut actions that haven't made change in environment.
        if self.detect_features and None in action.in_objects:
            prev_img = EnvironmentUtils.build_image_from_multilevel(self.multilevel, [])
        else:
            prev_img = last_image

        tool_status = action.tool.run_known_objects(self.multilevel.cur_env, arguments)

        if tool_status is False:
            return -1, False, False

        #print(tool_status)
        #if None in self.multilevel.cur_env.objs: #looks like this happens on empty intersection
            #print("err None inside objects: ")

        suc, new_objects = self.objects_based_on_img_diff(prev_img, action.tool, self.multilevel, arguments)

        for i in new_objects:
            self.all_objects.append(i)
        if suc is False:
            return -1, False, False
        if self.multilevel.cur_env.objs[-1] is not None and np.any(np.isnan(self.multilevel.cur_env.objs[-1].data)):
            return -1, False, False
        r, done = self.multilevel.evaluate_last_step(change_rem_goals=change_rem_goals)
        return r, done, True

    def initialize_search(self, multilevel, detector_model, dataset):
        tools = []
        for tool_name in multilevel.cur_env.enabled_tools:
            if tool_name == "move" or tool_name == "Point":
                continue
            tool_index = multilevel.tool_name_to_index[tool_name]
            tools.append([multilevel.tools[tool_index], tool_name])
        self.usable_tools = tools

        self.goal_size = len(self.multilevel.remaining_goals)

        if self.detect_features:
            image = EnvironmentUtils.build_image_from_multilevel(multilevel, [])
            image[:, :, 1] = 0
            detection = detector_model.detect([image], verbose=0, bool_masks=False)[0]
            visualize.display_instances(image, detection['rois'], detection['masks'] > 0.5, detection['class_ids'],
                                    dataset.class_names, detection['scores'],
                                    show_mask=True, show_bbox=True, ax=get_ax(), )

            all_objects, acc = DetectorUtils.find_environment_objects(detection, multilevel, dataset, include_goal=False)
        else:
            all_objects = list(self.multilevel.cur_env.visible_objs())
        return all_objects

    def deepening(self, multilevel, dataset, detector_model, level, output_to=sys.stdout, starting_depth=5, max_depth=10):
        self.multilevel = multilevel
        self.dataset = dataset
        self.detector_model = detector_model
        self.all_objects = self.initialize_search(multilevel, detector_model, dataset)
        number_of_geom_primitives = len(self.all_objects)
        sol = False
        for i in range(starting_depth, max_depth+1):
            self.branch_factor_sum = {}
            self.branch_level_visits = {}
            self.geom_primitives = {}
            self.cur_visits = 0
            self.cur_max_depth = i
            path, sol = self.search(0, [])

            if sol:
                break

        branching_factors = []
        estimate_branching_factors = []
        for i in range(len(self.branch_factor_sum)):
            if self.branch_factor_sum[i] == 0:
                break
            branching_factors.append("{:.2f}".format(self.branch_factor_sum[i] / self.branch_level_visits[i]))
            estimate_branching_factors.append("{}".format(TreeSearch.enumerate_estimate_of_branching_factor.
                                                       get_branching_factor(self.geom_primitives[i] / self.branch_level_visits[i], [i[1] for i in self.usable_tools])))

        print("{},{},{},{}".format(level, 'True' if sol else 'False', " - ".join(branching_factors)," - ".join(estimate_branching_factors)), file=output_to, flush=True)
        # for a in path: print(a)
        return path

    def search(self, depth, path, reward_sum=0):
        self.cur_visits += 1
        if self.cur_max_depth < depth or self.cur_visits > self.max_node_visits:
            return [], False
        s = State(self.multilevel, self.all_objects, build_image=self.detect_features)
        moves = self.get_all_moves(s, path, depth, reward_sum, detected_images=self.all_objects)

        if depth not in self.branch_factor_sum:
            self.branch_factor_sum[depth] = 0
            self.branch_level_visits[depth] = 0
            self.geom_primitives[depth] = 0
        self.branch_factor_sum[depth] += len(moves)
        self.branch_level_visits[depth] += 1
        self.geom_primitives[depth] += len(list(self.multilevel.cur_env.visible_objs()))

        for m in moves:
            if self.action_already_used(m, path):
                # do not use same action over again
                continue
            r, done, suc = self.execute_action(m, s.image)
            path.append(m)
            if done:
                self.reverse_to_state(s, self.all_objects, self.multilevel)
                return path, True
            res, sol = self.search(depth+1, path, reward_sum=reward_sum+r)
            if sol:
                self.reverse_to_state(s, self.all_objects, self.multilevel)
                return res, True
            self.reverse_to_state(s, self.all_objects, self.multilevel)
            path.pop()
        return [], False

    def get_all_moves(self, state, path, depth, reward_sum, detected_images=None):
        '''
        :param detected_images: when None this method will use model to detect objects in scene. If thi is an array with
            detected objects then we can skip detection and use those
        :return:
        '''
        image = state.image
        if detected_images is None:
            detection = self.detector_model.detect([image], verbose=0, bool_masks=False)[0]
            construction_indexes = detection["class_ids"] <= 3
            detection["masks"] = detection["masks"][:, :, construction_indexes]
            detection["class_ids"] = detection["class_ids"][construction_indexes]
            all_objects, acc = DetectorUtils.find_environment_objects(detection, self.multilevel, self.dataset,
                                                                      include_goal=False)
            visualize.display_instances(image, detection['rois'], detection['masks'] > 0.5, detection['class_ids'],
                                        self.dataset.class_names, detection['scores'],
                                        show_mask=True, show_bbox=True, ax=get_ax(), )
        else:
            all_objects = detected_images

        if self.use_hint:
            tool_id, _ = self.multilevel.get_construction(depth)
            tools = [[self.multilevel.tools[tool_id],self.multilevel.tool_index_to_name[tool_id]]]
        else:
            tools = self.usable_tools
        actions = get_all_possible_actions(all_objects, self.multilevel, self.dataset, tools, self.freedom_degrees)
        correct_actions = []
        for a in actions:
            if self.action_already_used(a, path):
                continue
            r, done, succes = self.execute_action(a, image, change_rem_goals=False)

            self.reverse_to_state(state, self.all_objects, self.multilevel)
            a.action_reward = r

            # Greedy for reward. This action should be in every solution (or atleast action with same output)
            if r > 0:
                return [a]

            # execute of the action makes no difference inside environment or some internal error has occurred.
            if succes and (self.cur_max_depth - depth) > ((1-reward_sum) * self.goal_size):
                correct_actions.append(a)

        actions = sorted(correct_actions, key=lambda x: x.action_reward, reverse=True)
        return actions

    def objects_based_on_img_diff(self, prev_image, tool, multilevel, args):
        '''
        Based on differnce between last 2 state images find new objects in scene, also return false if
        there is no diffrence which means we didn't draw anything new. just draw same geom primituve with different tool
        or based on different objects
        :param prev_image:
        :param tool:
        :param multilevel:
        :param args args used to run tool
        :return:
        '''
        out_t = tool.get_out_types(multilevel.cur_env, args)
        if not isinstance(out_t, (list, tuple)):
            out_t = [out_t]

        if not self.detect_features:
            # in this setup we don't have to detect features and we can get them straight from multilevel
            new_objects = []
            for i in range(len(out_t)):
                if not self.multilevel.cur_env.objs[-i-1] is None:
                    new_objects.append(self.multilevel.cur_env.objs[-i-1])
            return True, new_objects

        next_image = EnvironmentUtils.build_image_from_multilevel(self.multilevel, [])
        diff = np.array(next_image[:, :, 0] > 0, dtype=int) - np.array(prev_image[:, :, 0] > 0, dtype=int)
        # diff = next_image[:, :, 0] -  prev_image[:, :, 0]
        m = np.array(diff) > 0
        if not np.any(m):
            return False, []

        if len(out_t) > 1:
            # Should be only intersection tool and result should be [point, point]
            p1 = DetectorUtils.match(m, out_t[0], multilevel, include_goal=False)
            p1_c = [int(x) for x in p1.a / multilevel.scale]
            del_size = 5
            m[p1_c[0] - del_size:p1_c[0] + del_size, p1_c[1] - del_size:p1_c[1] + del_size] = np.zeros(
                (del_size * 2, del_size * 2))
            if not np.any(m):
                return True, [p1]
            p2 = DetectorUtils.match(m, out_t[1], multilevel, include_goal=False)
            return True, [p1, p2]
        else:
            return True, [DetectorUtils.match(m, out_t[0], multilevel, include_goal=False)]

