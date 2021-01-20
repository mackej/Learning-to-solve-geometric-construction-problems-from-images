import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import mrcnn.utils as utils
from collections import namedtuple


import matplotlib.pyplot as plt
from PIL import Image
from LevelSelector import LevelSelector
from collections import deque
import pickle
import matplotlib.pyplot as plt
from mrcnn import visualize
from GeometryObjectsDetector import DetectorGeometryDataset
import GeometryObjectsDetector.DetectorGeometryConfig as gconfig
import mrcnn.model as modellib
from py_euclidea import multi_level
import numpy as np
import enviroment_utils as env_utils
from GeometryObjectsDetector.DetectorUtils import DetectorUtils


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    """all paths to use:
    logs/just_line/mask_rcnn_geometryfromimages_0100.h5
    logs/circle/mask_rcnn_geometryfromimages_0100.h5
    logs/circle_heat_map/mask_rcnn_geometryfromimages_0100.h5
    """
    parser.add_argument("--model_path", default="logs_detector/detector01/mask_rcnn_detector_of_geom_primitives_0140.h5", type=str,
                        help="loads model. 'last' for last trained model in logs")
    parser.add_argument("--episodes", default=500, type=int,
                        help="how many episodes are used for evaluation for each environment.")

    parser.add_argument("--history_size", default=1, type=int,
                        help="history size")
    parser.add_argument("--visualization", default=1, type=int,
                        help="debug option ... visualisate single steps or not")
    parser.add_argument("--white_visualization", default=1, type=int,
                        help="generate visualisation with white background. It takes longer because for white"
                             "channel you cannot just merge channel as in black visualization.")
    parser.add_argument("--generate_levels", default="02.*10", type=str,
                        help="regex that matches lvl names")

    args = parser.parse_args()


    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    class InferenceConfig(gconfig.GeometryConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    def get_ax(rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(800/90, 800/90))
        return ax

    dataset = DetectorGeometryDataset.DetectorGeometry()
    dataset.PrepareDataGen(history_size=args.history_size, )
    dataset.prepare()

    inference_config = InferenceConfig()
    inference_config.IMAGE_META_SIZE += -inference_config.NUM_CLASSES + dataset.get_number_od_classes()
    inference_config.NUM_CLASSES = dataset.get_number_od_classes()
    #inference_config.NUM_CLASSES = dataset_train.get_number_od_classes()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    if args.model_path == "last":
        model_path = model.find_last()
    else:
        model_path = os.path.join(ROOT_DIR, args.model_path)

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    levels = LevelSelector.get_levels(match=args.generate_levels)
    m = multi_level.MultiLevel((
        levels
    ))
    history = deque(maxlen=args.history_size)
    accuracies = []
    number_of_possible_actions = len(m.tools)
    total_reward = 0
    env_solved = 0
    done = True
    accuracy = []
    for level in range(len(levels)):
        for e in range(args.episodes):
            level_index = m.next_level(level)
            env = m.cur_env
            # construct = construction(env, m.scale, number_of_possible_actions)
            reward_in_one_env = 0
            env_construction_len = m.get_construction_length()
            #reset history
            for i in range(args.history_size):
                history.append(np.zeros(m.out_size))

            for i in range(env_construction_len):

                image = env_utils.EnvironmentUtils.build_image_from_multilevel(m, history)
                pred = model.detect([image], verbose=0, bool_masks=False)[0]

                last_state = image[:, :, 0]
                history.append(last_state)

                # Uncoment to visualisate each prediction
                if args.visualization > 0:
                    visualization_scale = 1
                    caption_col = "white"
                    if args.white_visualization > 0:
                        visualization_scale = 4
                        vis_image = env_utils.EnvironmentUtils.build_image_from_multilevel_for_visualization(m, history, visualization_scale)
                        caption_col = "black"
                    else:
                        vis_image = image

                    visualize.display_instances(vis_image, pred['rois'], pred['masks'] > 0.5, pred['class_ids'],
                                                dataset.class_names, pred['scores'], caption_col=caption_col,
                                                show_mask=False, show_bbox=False, ax=get_ax(),upscale=visualization_scale)
                    plt.savefig("input_image{}.png".format(i), dpi=1024)
                    visualize.display_instances(vis_image, pred['rois'], pred['masks'] > 0.5, pred['class_ids'],
                                                dataset.class_names, pred['scores'], ax=get_ax(), caption_col=caption_col,upscale=visualization_scale)
                    plt.savefig("output_image{}.png".format(i), dpi=1024)

                _, acc = DetectorUtils.find_environment_objects(pred, m, dataset)
                accuracy.append(acc)
                construction_tool, construction_pts = m.get_construction(i)
                m.action_set_tool(construction_tool)
                for pt in construction_pts:
                    r, done, tool_status = m.action_click_point(pt, auto_proceed=False)
                    if tool_status is False and m.tool_mask[0]:
                        m.action_set_tool(0)
                        r, done, tool_status = m.action_click_point(pt)
                        break
                if done:
                    break
print('FINAL ACCURACY:', np.average(np.array(accuracy)))

