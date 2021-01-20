import os
import sys
from datetime import date

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from LevelSelector import LevelSelector
from collections import deque
import pickle
import matplotlib.pyplot as plt
from mrcnn import visualize

import MaskRCNNExperiment.GeometryConfig as gconfig
from MaskRCNNExperiment.SingleModelInference import SingleModelInference
from MaskRCNNExperiment.MultipleModelsInference import MultipleModelsInference
from MaskRCNNExperiment.InteractiveInferenceModel import *
import mrcnn.model as modellib
from py_euclidea import multi_level
import numpy as np
import enviroment_utils as env_utils

class unfinished_environment:
    def __init__(self, objs, goal, level_index, construction):
        self.objs = objs
        self.goal = goal
        self.level_index = level_index
        self.construction = construction


def TestModel(levels, model_path, episodes,  history_size,
              additional_moves, use_hint=False,
              loop_failed=False, failed_envs=[],
              visualization=False, white_visualization=False,
              log_fails=False, log_failed_levels_file=None, output_to=sys.stdout,
              model_type="SingleMofdelInference"):

    levels = LevelSelector.get_levels(levels)

    model_types = {
        "MultipleModelsInference": MultipleModelsInference,
        "SingleModelInference": SingleModelInference,
        "InteractiveInferenceSingleModel": InteractiveInferenceSingleModel,
        "InteractiveInferenceMultiModel": InteractiveInferenceMultiModel
    }

    model = model_types[model_type].load_model(model_path, output_to=output_to)

    unfinished_envs = []

    m = multi_level.MultiLevel((
        levels
    ))
    history = deque(maxlen=history_size)
    accuracies = []
    number_of_possible_actions = len(m.tools)
    total_reward = 0
    env_solved = 0
    done = True

    if not os.path.exists(log_failed_levels_file):
        os.makedirs(log_failed_levels_file)

    for level in range(len(levels)):
        for e in range(episodes):
            if loop_failed:
                level_index = m.next_level(None, failed_envs[e])
            else:
                level_index = m.next_level(level)

            # construct = construction(env, m.scale, number_of_possible_actions)
            reward_in_one_env = 0
            env_construction_len = m.get_construction_length()
            # reset history
            for i in range(history_size):
                history.append(np.zeros(m.out_size))

            for i in range(env_construction_len + additional_moves):

                image = env_utils.EnvironmentUtils.build_image_from_multilevel(m, history)
                pred = model.detect([image], verbose=0, bool_masks=False)

                last_state = image[:, :, 0]
                history.append(last_state)

                # visualisation of inference
                if visualization:
                    visualization_scale = 1
                    caption_col = "white"
                    if white_visualization:
                        visualization_scale = 4
                        vis_image = env_utils.EnvironmentUtils.build_image_from_multilevel_for_visualization(m, history,
                                                                                                             visualization_scale)
                        caption_col = "black"
                    else:
                        vis_image = image

                    model.show_model_input(vis_image, pred, caption_col, visualization_scale, get_ax())
                    #plt.savefig("Outputimgs/input_image{}.png".format(i), dpi=1024)
                    model.show_model_output(vis_image, pred, caption_col, visualization_scale, get_ax())
                    #plt.savefig("Outputimgs/output_image{}.png".format(i), dpi=1024)


                hint, _ = [None, None] if not use_hint else m.get_construction(i)
                r, done = model.inference_step(pred, hint, m)
                if r is None and done is None:
                    break

                reward_in_one_env += r
                if done:
                    if visualization and white_visualization:
                        visualization_scale = 4
                        vis_image = env_utils.EnvironmentUtils.build_image_from_multilevel_for_visualization(m, history,
                                                                                                             visualization_scale)
                        model.show_model_input(vis_image, pred, caption_col, visualization_scale, get_ax())
                        plt.savefig("Outputimgs/input_image{}.png".format(i + 1), dpi=1024)
                    break
            if not done and log_fails:
                m.cur_env.restart()
                objs_copy = [ob.copy() for ob in m.cur_env.objs]
                goals_copy = [g.copy() for g in m.cur_env.cur_goal()]
                unfinished_envs.append(unfinished_environment(objs_copy, goals_copy, level_index,
                                           m.cur_env.construction))

            if reward_in_one_env == 1.0:
                env_solved += 1

            total_reward += reward_in_one_env

        print("level", levels[level][1], "accuracy: ", total_reward, "/", e + 1, "total accuracy:",
              total_reward / (e + 1),
              " solved envs:", env_solved, "/", e + 1, file=output_to, flush=True)

        accuracies.append(total_reward / (e + 1))
        total_reward = 0
        env_solved = 0

        if len(unfinished_envs) != 0 and log_fails:
            file = os.path.join(log_failed_levels_file, levels[level][0]+levels[level][1])
            with open(file, "wb") as data_gen_file:
                pickle.dump(unfinished_envs, data_gen_file)
            unfinished_envs = []
    csv_output = open(os.path.join(ROOT_DIR, os.path.join(os.path.dirname(model_path),
                                                          "output_{}.csv".format(date.today()))), "w")
    print(", ".join([l[1] for l in levels]), file=csv_output)
    print(", ".join([str(i) for i in accuracies]), file=csv_output, flush=True)
    print('FINAL ACCURACY:', np.average(np.array(accuracies)), file=output_to, flush=True)

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(800/90, 800/90))
    return ax


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    # logs/Beta_as_whole;logs/Gamma_as_whole;logs/redoing_alpha
    parser.add_argument("--model_path", default="logs/All_at_once/20201016/mask_rcnn_geometryfromimages_0400.h5", type=str,
                        help="Loads the model from given path, use 'last' for last trained model in logs")
    parser.add_argument("--episodes", default=1, type=int,
                        help="How many episodes are used for evaluation")
    parser.add_argument("--hint", default=0, type=int,
                        help="Get hint which tool should be used each time. 1 use hint, 0 do not")
    parser.add_argument("--additional_moves", default=2, type=int,
                        help="How much more moves evaluation gets on top of minimal construction length.")
    parser.add_argument("--log_failed_levels", default=0, type=int,
                        help="'1' if failed levels should be saved, '0' if not.")
    parser.add_argument("--log_failed_levels_file", default="../logs/Epsilon_One_by_One/20200907/05-06_both_goals/unfinished_levels/py_euclidea.05_epsilon06_Hash", type=str,
                        help="From this file will be loaded unfinished levels.")
    parser.add_argument("--load_levels_from_failed_logs", default=0, type=int,
                        help="'1' if the program should load unfinished level file, '0' otherwise")
    parser.add_argument("--history_size", default=1, type=int,
                        help="history size")
    parser.add_argument("--visualization", default=1, type=int,
                        help="'1' if each step should be visualized, '0' otherwise.")
    parser.add_argument("--white_visualization", default=1, type=int,
                        help="generate visualisation with white background. It takes longer because for white"
                             "channel you cannot just merge channel as in black visualization. '1' on, '0' off")
    parser.add_argument("--generate_levels", default="05.*10", type=str,
                        help="Regular expresion matching levels to generate")
    parser.add_argument("--tool_set", default="min_by_construction", type=str,
                        help="\"min_by_levels\" to generate minimal set of tools given by level;"
                             "\"min_by_construction\" to generate minimal set of tools given by construction;"
                             " Other values for all tools")
    parser.add_argument("--model_type", default="SingleModelInference", type=str,
                        choices=["MultipleModelsInference", "SingleModelInference", "InteractiveInferenceSingleModel", "InteractiveInferenceMultiModel"],
                        help="specify which type of model should be used."
                             " U also need to set proper --model_path based on model type")

    args = parser.parse_args()

    use_hint = args.hint > 0
    loop_failed = args.load_levels_from_failed_logs > 0
    failed_envs = []
    if loop_failed:
        with open(args.log_failed_levels_file, "rb") as file:
            failed_envs = pickle.load(file)
    log_fails = args.log_failed_levels > 0

    TestModel(args.generate_levels, args.model_path, args.episodes, args.history_size, args.additional_moves,
              use_hint=use_hint, loop_failed=loop_failed, failed_envs=failed_envs,
              visualization=args.visualization > 0, white_visualization=args.white_visualization >0,
              log_fails=log_fails, log_failed_levels_file=args.log_failed_levels_file,
              model_type=args.model_type)

