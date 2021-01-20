from MaskRCNNExperiment.InteractiveInferenceModel import *
from LevelSelector import *
from py_euclidea.multi_level import *
from collections import deque
import time
from TreeSearch.enumerate_estimate_of_branching_factor import get_branching_factor
from PIL import Image

class HypohesisTreeSearch:

    def __init__(self, models, levels, epoch_each_level = 500, history_size=1, model_disabling=True):
        self.model = InteractiveInferenceMultiModel.load_model_exact_path([os.path.join("logs",i) for i in models])
        self.levels = levels
        self.epochs = epoch_each_level
        self.history_size = history_size
        self.max_depth = None
        self.visits = None
        self.branching_factors_sum = None
        self.time_elapses = None
        self.used_models = {}
        self.model_disabling = model_disabling
        self.max_visits = 500
    def test_all_levels(self):
        result = {}
        print("level, accuracy, visists, branching factor, estimates, time, time min_max, used models")
        for i in range(len(self.levels)):
            self.used_models = {}
            if self.model_disabling:
                self.model.enabled[i] = False

            level_to_test = [self.levels[i]]
            accuracy = self.test_inference(level_to_test)

            result[self.levels[i][1]] = accuracy / self.epochs
            elapsed = np.array(self.time_elapses)
            branching_factors = []
            estimate_branching_factors = []
            for j in range(len(self.branching_factors_sum)):
                if self.branching_factors_sum[j] == 0:
                    break
                branching_factors.append("{:.2f}".format(self.branching_factors_sum[j] / self.visits[j]))
                estimate_branching_factors.append("{}".format(
                                                              get_branching_factor(self.number_of_geom_primitives + j,
                                                                                   [j[1] for j in self.usable_tools])))
            print("{},{},{},{},{},{},{},{}".format(
                str(self.levels[i][0] + self.levels[i][1] ).replace('_', ' '),
                "{:.2f}".format(accuracy / self.epochs),
                np.sum(np.array(self.visits)),
                " - ".join(branching_factors),
                " - ".join(estimate_branching_factors),
                "{:.2f}".format(np.average(elapsed)),
                "{:.2f} / {:.2f}".format(np.min(elapsed), np.max(elapsed)),
                str(self.used_models).replace(',', ';')
            ), flush=True)
            if self.model_disabling:
                self.model.enabled[i] = True
        return result

    def test_inference(self, level_to_test):
        m = MultiLevel(level_to_test)
        m.next_level()
        self.number_of_geom_primitives = len(m.cur_env.objs)
        tools = []
        for tool_name in m.cur_env.enabled_tools:
            if tool_name == "move" or tool_name == "Point":
                continue
            tool_index = m.tool_name_to_index[tool_name]
            tools.append([m.tools[tool_index], tool_name])
        self.usable_tools = tools
        self.max_depth = m.get_construction_length()
        self.visits = [0] * (self.max_depth+1)
        self.time_elapses = []
        self.branching_factors_sum = [0] * (self.max_depth+1)
        solved_levels = 0
        for i in range(self.epochs):
            self.history = deque()
            for i in range(self.history_size):
                self.history.append(np.zeros(m.out_size))
            start = time.time()
            self.one_iter_visits = 0
            if self.solve(m):
                solved_levels += 1
            end = time.time()
            self.time_elapses.append((end - start))
            m.next_level()
        return solved_levels

    def get_history(self):
        return list(self.history)[-self.history_size:]

    def solve(self, multi_level, depth=0):
        self.one_iter_visits +=1
        if depth > self.max_depth or self.max_visits < self.one_iter_visits:
            return False
        self.visits[depth] += 1
        image = env_utils.EnvironmentUtils.build_image_from_multilevel(multi_level, self.get_history())
        #Image.fromarray(image).show()
        pred = self.model.detect([image], verbose=0, bool_masks=False)
        self.history.append(image[:, :, 0])
        hypoheses = self.model.gather_hypotheses(pred, None, multi_level)
        # if we have some reward we should go for it
        if len(hypoheses)!=0 and hypoheses[0]["reward"] > 0:
            hypoheses = [hypoheses[0]]

        self.branching_factors_sum[depth] += len(hypoheses)
        for h in hypoheses:
            r, done = self.run_step(h, multi_level)
            if done:
                if h["source"] not in self.used_models:
                    self.used_models[h["source"]] = 0
                self.used_models[h["source"]] += 1
                return True
            done = self.solve(multi_level, depth=depth+1)
            if done:
                if h["source"] not in self.used_models:
                    self.used_models[h["source"]] = 0
                self.used_models[h["source"]] += 1
                return True
            self.revese_step(h, multi_level)
        self.history.pop()
        return False

    def run_step(self, h, multilevel):
        for s in h["steps"]:
            multilevel.cur_env.add_and_run(s)
        r, done = multilevel.evaluate_last_step()
        return r, done

    def revese_step(self, h, multi_level):
        for _ in h["steps"]:
            multi_level.cur_env.pop()

    def print_enabled_model(self):
        for i in range(len(self.model.model)):
            m = self.model.model[i]
            print("{} - {}".format(m.model_path, ("Enabled" if self.model.enabled[i] else "Disabled")))
        print('\n')

