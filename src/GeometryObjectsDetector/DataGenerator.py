from mrcnn import visualize
from py_euclidea import multi_level
from GeometryObjectsDetector import DetectorGeometryDataset
from PIL import Image
from LevelSelector import LevelSelector

def generate_geometry_episodes(args):
    data_gen = DetectorGeometryDataset.DetectorGeometry()
    data_gen.PrepareDataGen(mask_size=args.mask_size,
                            history_size=args.history_size,)

    data_gen_val = DetectorGeometryDataset.DetectorGeometry()
    data_gen_val.PrepareDataGen(mask_size=args.mask_size,
                                history_size=args.history_size)
    levels = LevelSelector.get_levels(
        match=args.generate_levels
    )
    m = multi_level.MultiLevel((levels))
    number_of_possible_actions = len(m.tools)

    dataset = data_gen
    progress = 0
    print("data generation: start")
    for epoch in range(args.train_epochs + args.val_epochs):
        if epoch != 0 and int((args.train_epochs + args.val_epochs) / (100 / 5)) != 0 and\
                epoch % int((args.train_epochs + args.val_epochs) / (100 / 5)) == 0:
            progress += 5
            print(progress, "% done")
        level_index = m.next_level()

        env = m.cur_env

        for action_index in range(m.get_construction_length()):

            action_tool, action_points = m.get_construction(action_index)
            m.action_set_tool(action_tool)
            primitives = env.get_copy_of_state_objects()

            if epoch == args.train_epochs - 1:
                dataset = data_gen_val

            dataset.add_one_scene(primitives, m, epoch)

            if args.visualize > 0:
                # img = Image.fromarray(EnvironmentUtils.build_image_from_multilevel(m)).show()
                img = data_gen.load_image(len(data_gen.image_info) - 1)
                # Image.fromarray(img).show()
                masks, indcs = dataset.load_mask(len(dataset.image_info) - 1)
                visualize.display_top_masks(img, masks, indcs, list(dataset.id_name_dic.keys()))
            try:
                for pt in action_points:
                    r, done, tool_status = m.action_click_point(pt, auto_proceed=False)
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
    print("data generation: done")
    return data_gen, data_gen_val
