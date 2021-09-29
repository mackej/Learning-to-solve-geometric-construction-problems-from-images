from MaskRCNNExperiment.MultipleModelsInference import *
from MaskRCNNExperiment.SingleModelInference import *
from enviroment_utils import *
import math
import matplotlib.pyplot as plt


class InteractiveInference:
    def inference_step(self, pred, hint, multi_level):
        all_hypothesis = self.gather_hypotheses(pred, hint, multi_level)
        all_hypothesis.sort(key=lambda x: -x["score"])
        if len(all_hypothesis) == 0:
            return None, None
        visualization_scale = 4

        fig = plt.figure(figsize=(8, 8))
        columns = 2
        rows = math.ceil(len(all_hypothesis) / columns)

        for i in range(len(all_hypothesis)):
            h = all_hypothesis[i]
            h_objs = h["result_obj"]
            h_inputs = []
            for s in h["steps"][-1].args_i:
                if s < len(multi_level.cur_env.objs):
                    h_inputs.append(multi_level.cur_env.objs[s])
            img = env_utils.EnvironmentUtils.build_image_from_multilevel_for_visualization(multi_level, [],
                                                                                           visualization_scale,
                                                                                           highlight_objects=h_objs,
                                                                                           higlight_inputs=h_inputs)
            ax = fig.add_subplot(rows, columns, i + 1)
            name = "{}({:.3f}) - {}".format(multi_level.tool_index_to_name[all_hypothesis[i]["action_id"]], h["score"],
                                            str(i))
            ax.set_title(name)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            plt.imshow(img)

        plt.show()
        print("choose and action [0-{}]".format(len(all_hypothesis) - 1))
        ch = int(input())
        plt.close()

        chosen_hypothesis = all_hypothesis[ch]
        for s in chosen_hypothesis["steps"]:
            multi_level.cur_env.add_and_run(s)
        r, done = multi_level.evaluate_last_step(size=len(chosen_hypothesis["steps"]))
        return r, done
    def build_hypothesis_visualisations(self, pred, hint, multi_level):
        all_hypothesis = self.gather_hypotheses(pred, hint, multi_level)

        if len(all_hypothesis) == 0:
            return None
        visualization_scale = 4
        res = []
        for i in range(len(all_hypothesis)):
            h = all_hypothesis[i]
            h_objs = h["result_obj"]
            h_inputs = []
            for s in h["steps"][-1].args_i:
                if s < len(multi_level.cur_env.objs):
                    h_inputs.append(multi_level.cur_env.objs[s])
            img = env_utils.EnvironmentUtils.build_image_from_multilevel_for_visualization(multi_level, [],
                                                                                           visualization_scale,
                                                                                           highlight_objects=h_objs,
                                                                                           higlight_inputs=h_inputs)
            res.append({"image": img, "hypothesis": all_hypothesis[i],
                        "tool_name": multi_level.tool_index_to_name[all_hypothesis[i]["action_id"]]})
        return res

class InteractiveInferenceMultiModel(InteractiveInference, MultipleModelsInference):
    @staticmethod
    def gather_hypotheses(pred, hint, multi_level, datasets, models_paths, cut_cheat_moves=False):
        all_hypothesis = []
        for i in range(len(pred)):
            p = pred[i]
            if p is None:
                continue
            h = env_utils.EnvironmentUtils.prepare_all_hypothesis(p, datasets[i], multi_level)
            for j in h:
                if j["tool_status"]:
                    j["source"] = models_paths[i]
                    all_hypothesis.append(j)
        unique_hypothesis = []
        for i in range(len(all_hypothesis)):
            duplicate = False
            for j in range(i+1, len(all_hypothesis)):
                if EnvironmentUtils.compare_hypothesis(all_hypothesis[i], all_hypothesis[j]):
                    duplicate = True
                    break
            if not duplicate:
                unique_hypothesis.append(all_hypothesis[i])
        unique_hypothesis.sort(reverse=True, key=lambda x: (x["reward"], x["score"]))
        return unique_hypothesis


class InteractiveInferenceSingleModel(InteractiveInference, SingleModelInference):
    def gather_hypotheses(self, pred, hint, multi_level):
        return env_utils.EnvironmentUtils.prepare_all_hypothesis(pred, self.dataset, multi_level)

