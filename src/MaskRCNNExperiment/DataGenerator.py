from mrcnn import visualize
from py_euclidea import multi_level
from MaskRCNNExperiment import GeometryDataset
from PIL import Image
from LevelSelector import LevelSelector
import enviroment_utils as env_utils
from mrcnn import utils
import numpy as np
import matplotlib.pyplot as plt

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(800/90, 800/90))
    return ax

def generate_geometry_episodes(args):
    levels = LevelSelector.get_levels(
        match=args.generate_levels
    )
    m = multi_level.MultiLevel((levels))
    min_tool_set = None
    if args.tool_set == "min_by_levels":
        min_tool_set = m.get_min_set_of_tool()
    if args.tool_set == "min_by_construction":
        min_tool_set = m.get_min_set_of_tools_for_constructions()

    data_gen = GeometryDataset.GeometryDataset()
    data_gen.PrepareDataGen(use_heat_map=args.use_heat_map>0, heat_map_size=args.mask_size,
                            history_size=args.history_size, heat_map_cov=args.heat_map_covariance,
                            tool_list=min_tool_set)

    data_gen_val = GeometryDataset.GeometryDataset()
    data_gen_val.PrepareDataGen(use_heat_map=args.use_heat_map>0, heat_map_size=args.mask_size,
                                history_size=args.history_size, heat_map_cov=args.heat_map_covariance,
                                tool_list=min_tool_set)


    number_of_possible_actions = len(m.tools)

    dataset = data_gen
    progress = 0
    print("data generation: start", flush=True)
    for epoch in range(args.train_epochs + args.val_epochs):
        if epoch != 0 and int((args.train_epochs + args.val_epochs) / (100 / 5)) != 0 and\
                epoch % int((args.train_epochs + args.val_epochs) / (100 / 5)) == 0:
            progress += 5
            print(progress, "% done", flush=True)
        level_index = m.next_level()

        env = m.cur_env

        for action_index in range(m.get_construction_length()):

            action_tool, action_points = m.get_construction(action_index)
            m.action_set_tool(action_tool)
            action_tool_network_index = data_gen.id_name_dic[m.tool_index_to_name[action_tool]]
            primitives = env.get_copy_of_state_objects()

            if epoch == args.train_epochs - 1:
                dataset = data_gen_val

            dataset.add_one_scene(primitives, action_points.copy(), action_tool_network_index, m, epoch)

            if args.visualize > 0 and action_index == 0:
                # img = Image.fromarray(EnvironmentUtils.build_image_from_multilevel(m)).show()
                img = data_gen.load_image(len(data_gen.image_info) - 1)
                vis_image = env_utils.EnvironmentUtils.build_image_from_multilevel_for_visualization(m, None,
                                                                                                     4)
                #Image.fromarray(vis_image).show()

                masks, indcs = dataset.load_mask(len(dataset.image_info) - 1)
                bbox = utils.extract_bboxes(masks)
                im = visualize.display_top_masks(vis_image, masks, indcs, list(dataset.id_name_dic.keys()))
                im.close()
                #following line were used to generate example training data for the thesis.
                #visualize.display_instances(vis_image, bbox[1:, :], masks[:, :, 1:], indcs[1:],
                 #                           [i['name'] for i in data_gen.class_info], [None] * len(indcs),
                  #                          caption_col='black', upscale=4, ax=get_ax(),
                   #                         colors=((0.5, 0, 0.5), (0.5, 0.5, 0.0), (0, 0.5, 0.5)))
                #plt.savefig("Outputimgs/input_image{}_{}_{}.png".format(epoch, action_index+1,1), dpi=1024)
                #plt.close()
                #visualize.display_instances(vis_image, bbox[0:1, :], masks[:, :, 0:1], indcs[0:1],
                 #                           [i['name'] for i in data_gen.class_info], [None] * len(indcs),
                  #                          caption_col='black', upscale=4, ax=get_ax(), colors=((0.5, 0, 0.5), (0.5, 0.5, 0.0), (0, 0.5, 0.5)))
                #plt.savefig("Outputimgs/input_image{}_{}_{}.png".format(epoch, action_index + 1, 2), dpi=1024)
                #plt.close()
                #visualize.display_instances(vis_image, bbox[0:1, :], masks[:, :, 0:1], indcs[0:1],
                 #                           [i['name'] for i in data_gen.class_info], [None] * len(indcs),
                  #                          caption_col='black', upscale=4, ax=get_ax(), show_mask = False, show_bbox = False,
                   #                         colors=((0.5, 0, 0.5), (0.5, 0.5, 0.0), (0, 0.5, 0.5)))
                #plt.savefig("Outputimgs/input_image{}_{}_{}.png".format(epoch, action_index + 1, 0), dpi=1024)


            try:
                for pt in action_points:
                    # adding +0.1 so we'r not clicking perfectly, so while generation cant happen
                    # we just click on some object that is just in goal.
                    r, done, tool_status = m.action_click_point(pt+0.1, auto_proceed=False)
                    if tool_status is False and m.tool_mask[0]:
                        m.action_set_tool(0)
                        r, done, tool_status = m.action_click_point(pt)
                        break
            except Exception as e:
                img = dataset.load_image(len(dataset.image_info) - 1)
                masks, indcs = dataset.load_mask(len(dataset.image_info) - 1)
                visualize.display_top_masks(img, masks, indcs, list(dataset.id_name_dic.keys()))
                raise e

        if len(m.remaining_goals) != 0:
            s = m.get_state()
            d = s[:, :, [1, 2]]
            img = Image.fromarray(s[:, :, [1, 2, 3]]).show()
            raise Exception('Unfinished environment ' + levels[level_index][0] + "." + levels[level_index][1])

    data_gen_val.prepare()
    data_gen.prepare()
    print("data generation: done", flush=True)
    return data_gen, data_gen_val
