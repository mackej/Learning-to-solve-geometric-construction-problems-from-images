import TreeSearch.IterativeDeepening as itd

from MaskRCNNExperiment.SingleModelInference import SingleModelInference
from TreeSearch.State import *
import enviroment_utils

class ITD_with_MRCNN_model(itd.IterativeDeepening):
    def __init__(self, path_to_mrccn_model, starting_depth=5, max_depth=10, detect_features=True, tool_hints=False):
        model = SingleModelInference.load_model(path_to_mrccn_model)
        self.mrcnn_model = model.model
        self.mrcnn_dataset = model.dataset
        self.mrcnn_config = model.config

        super().__init__(starting_depth=starting_depth, max_depth=max_depth, detect_features=detect_features,
                       tool_hints=tool_hints)

    def deepening(self, multilevel, dataset, detector_model):

        super().deepening(multilevel, dataset, detector_model)

    def unifi_action_with_mrcnn_action(self, moves, mrcnn_tool, mrcnn_clicks):
        mrcnn_tool_name = self.multilevel.tool_index_to_name[mrcnn_tool]
        for i in range(len(mrcnn_clicks)):
            self.multi_level.cur_env.closest_obj_of_type(mrcnn_clicks[i],self.multilevel.tools[mrcnn_tool].in_types[i])

        for move in moves:
            if move.tool_name != mrcnn_tool_name:
                continue




    def search(self, depth, path, reward_sum=0):
        if self.cur_max_depth < depth:
            return [], False
        s = State(self.multilevel, self.all_objects, build_image=True)
        moves = self.get_all_moves(s, path, depth, reward_sum, detected_images=self.all_objects)
        if moves[0].action_reward !=0:
            pred = self.mrcnn_model.detect([s.image], verbose=0, bool_masks=False)[0]
            action, points, pred_index = EnvironmentUtils.action_from_prediction(pred, self.mrcnn_dataset, self.multilevel.tool_name_to_index)
            index = self.unifi_action_with_mrcnn_action(moves, action, points)

        self.branch_factor_sum[depth] += len(moves)
        self.branch_level_visits[depth] += 1

        for m in moves:
            if self.action_already_used(m, path):
                # do not use same action over again
                continue
            r, done, suc = self.execute_action(m, s.image)
            path.append(m)
            if done:
                self.reverse_to_state(s, self.all_objects, self.multilevel)
                return path, True
            res, sol = self.search(depth + 1, path, reward_sum=reward_sum + r)
            if sol:
                self.reverse_to_state(s, self.all_objects, self.multilevel)
                return res, True
            self.reverse_to_state(s, self.all_objects, self.multilevel)
            path.pop()
        return [], False
