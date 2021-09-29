import os, sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
#from skimage import draw
import MaskRCNNExperiment.GeometryDataset_on_the_fly_gen as euclidea_datagen
#import YolactExperiment.data.coco as cocodataset

from YolactExperiment.utils.augmentations import *
import numpy as np
import torch
import torch.utils.data as data
from enviroment_utils import EnvironmentUtils as env_utils

class Yolact_datagen(data.Dataset):

    def __init__(self, args):
        self.euclidea = euclidea_datagen.GeometryDataset_on_the_fly_gen(args)
        self.transform = BaseTransform()
        self.len = args.epoch_size
        self.name= "euclidea"
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)
    def pull_item(self, index):
        index = int(index)
        img, masks, classes = self.load_image_and_mask(index)

        #vizualization: to check images before
        #from mrcnn import visualize
        #im = visualize.display_top_masks(img, masks, classes, list(self.euclidea.id_name_dic.keys()))
        #im.close()

        #derive bboxes
        bboxes = self.extract_bboxes(masks)
        # IS it LIKE this or 2,1,0 ???
        masks = masks.transpose((2, 0, 1))
        num_crowds = 0
        height, width, _ = img.shape
        scale = np.array([width, height, width, height])
        target = []
        for i in range(len(classes)):
            bbox = bboxes[i]
            final_box = list(bbox/scale)
            final_box.append(classes[i])
            target.append(final_box)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                                           {'num_crowds': 0, 'labels': classes})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels = labels['labels']

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds

    def load_image_and_mask(self, image_id):
        self.euclidea.number_of_images += 1
        self.euclidea.step_index += 1
        if self.euclidea.done_level:
            self.euclidea.reset_level()

        action_tool, action_points = self.euclidea.multi_level.get_construction(self.euclidea.step_index)

        if action_tool is None and action_points is None:
            print("error in generation")
            self.euclidea.reset_level()
            self.euclidea.number_of_errors += 1
            action_tool, action_points = self.euclidea.multi_level.get_construction(self.euclidea.step_index)

        action_tool_network_index = self.euclidea.id_name_dic[self.euclidea.multi_level.tool_index_to_name[action_tool]]

        img = env_utils.build_image_from_multilevel(self.euclidea.multi_level, self.euclidea.history)
        self.euclidea.history.append(img[:, :, 0])
        self.euclidea.execute_action(action_tool, action_points)
        masks, classes = self.euclidea.procces_mask(action_points, action_tool_network_index)

        return img, masks, classes

    @staticmethod
    def extract_bboxes(masks):
        boxes = np.zeros([masks.shape[-1], 4], dtype=np.int32)
        for i in range(masks.shape[-1]):
            m = masks[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            #print("np.any(m, axis=0)", np.any(m, axis=0))
            #print("p.where(np.any(m, axis=0))", np.where(np.any(m, axis=0)))
            vertical_indicies = np.where(np.any(m, axis=1))[0]

            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x2 += 1
                y2 += 1
            else:
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])



        return boxes.astype(np.int32)
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str