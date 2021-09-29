import os
import sys
import time
os.path.abspath("../")
from MaskRCNNExperiment.SingleModelInference import *
from yolact import Yolact
from utils.augmentations import BaseTransform, BaseTransformVideo, FastBaseTransform, Resize

from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
from utils.tensorrt import convert_to_tensorrt

from data import cfg, set_cfg, set_dataset

import logging

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

class args_container:
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key,value)

class YolactSingleModelInference(SingleModelInference):
    def __init__(self, net, args, dataset):
        self.net = net
        self.args = args
        self.dataset = dataset
        self.sum_elapsed = 0
        self.elapsed_cnt = 0
    def detect(self, images, verbose=0, bool_masks=True):
        t_start = time.time()
        with torch.no_grad():
            img = images[0]
            #frame = torch.from_numpy(img).cuda().float()
            img = BaseTransform()(img)[0]
            img = torch.from_numpy(img).permute(2, 0, 1)
            batch = Variable(img.unsqueeze(0))
            if self.args.cuda:
                batch = batch.cuda()

            extras = {"backbone": "full", "interrupt": False,
                  "moving_statistics": {"aligned_feats": []}}
            preds = self.net(batch, extras=extras)["pred_outs"]

            _, _, masks, classes, scores, boxes = self.prep_display(preds, img, 256, 256,binarize_masks=False)
            #inverse permutation 2,0,1
            if len(classes)!=0:
                masks = masks.permute(1, 2, 0).detach().cpu().numpy()
                #boxes = boxes[:,[1,0,3,2]]
                boxes = boxes[:, [1, 0, 3, 2]]
        t_end = time.time() - t_start
        self.sum_elapsed += t_end
        self.elapsed_cnt += 1
        return {'masks':masks, 'class_ids':classes, 'scores':scores, 'rois':boxes}
    @staticmethod
    def load_model(model_path, output_to=sys.stdout):
        alternative_dataset = os.path.abspath(
            os.path.dirname(model_path) + os.path.sep + "GeometryDataset.py")
        if os.path.exists(alternative_dataset):
            # if model file have its special dataset... then load it instead of default one
            spec = importlib.util.spec_from_file_location("alternative_dataset", alternative_dataset)
            ds = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ds)
            dataset = ds.GeometryDataset()

        else:
            from MaskRCNNExperiment import GeometryDataset

            dataset = GeometryDataset.GeometryDataset()
        dataset.PrepareDataGen(tool_list_file=os.path.dirname(model_path))
        # dataset.PrepareDataGen()
        dataset.prepare()


        #we use same arguments as default parametters as in eval.py to find out which parammeter does what look at eval arg parser
        args = args_container(ap_data_file='results/ap_data.pkl', bbox_det_file='results/bbox_detections.json', benchmark=False, calib_images=None, coco_transfer=False, config=None, crop=True, cuda=True, dataset=None, detect=False, deterministic=False, disable_tensorrt=True, display=True, display_bboxes=True, display_lincomb=False, display_masks=True, display_scores=True, display_text=True, drop_weights=None, epoch_size=100000, eval_stride=5, fast_eval=False, fast_nms=False, generate_levels='alpha.*', heat_map_covariance=100, history_size=1, image=None, images=None, mask_det_file='results/mask_detections.json', mask_proto_debug=False, mask_size=5, max_images=-1, no_bar=False, no_hash=False, no_sort=True, output_coco_json=False, output_web_json=False, resume=False, score_threshold=0.01, seed=None, shuffle=False, tool_set='min_by_construction', top_k=100, trained_model='weights/resnet_test_3_bboxes/yolact_edge_28_180000.pth', trt_batch_size=1, use_fp16_tensorrt=False, use_heat_map=0, use_tensorrt_safe_mode=False, video=None, video_multiframe=1, web_det_path='web/dets/', yolact_transfer=False)
        args.trained_model = model_path
        args.score_threshold = 0.2
        args.top_k = 100

        if args.config is not None:
            set_cfg(args.config)

        if args.trained_model == 'interrupt':
            args.trained_model = SavePath.get_interrupt('weights/')
        elif args.trained_model == 'latest':
            args.trained_model = SavePath.get_latest('weights/', cfg.name)

        if args.config is None:
            model_path = SavePath.from_str(args.trained_model)
            # TODO: Bad practice? Probably want to do a name lookup instead.
            args.config = model_path.model_name + '_config'
            print('Config not specified. Parsed %s from the file name.\n' % args.config)
            set_cfg(args.config)

        if args.detect:
            cfg.eval_mask_branch = False

        if args.dataset is not None:
            set_dataset(args.dataset)

        from utils.logging_helper import setup_logger
        setup_logger(logging_level=logging.INFO)
        logger = logging.getLogger("yolact.eval")

        with torch.no_grad():
            if not os.path.exists('results'):
                os.makedirs('results')

            if args.cuda:
                cudnn.benchmark = True
                cudnn.fastest = True
                if args.deterministic:
                    cudnn.deterministic = True
                    cudnn.benchmark = False
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            #if args.resume and not args.display:
               # with open(args.ap_data_file, 'rb') as f:
               #     ap_data = pickle.load(f)
               # calc_map(ap_data)
               # exit()

            # if args.image is None and args.video is None and args.images is None:
            #   if cfg.dataset.name == 'YouTube VIS':
            #      dataset = YoutubeVIS(image_path=cfg.dataset.valid_images,
            #                              info_file=cfg.dataset.valid_info,
            #                             configs=cfg.dataset,
            #                            transform=BaseTransformVideo(MEANS), has_gt=cfg.dataset.has_gt)
            # else:
            #   dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
            #               transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            # prep_coco_cats()
            # else:
            #            dataset = None
            #dataset = euclidea_dataset.Yolact_datagen(args)
            #cfg.dataset.class_names = dataset.euclidea.class_names
            logger.info('Loading model...')
            net = Yolact(training=False)
            if args.trained_model is not None:
                net.load_weights(args.trained_model, args=args)
            else:
                logger.warning("No weights loaded!")
            net.eval()
            logger.info('Model loaded.')

            convert_to_tensorrt(net, cfg, args, transform=BaseTransform())

            if args.cuda:
                net = net.cuda()
            net.detect.use_fast_nms = args.fast_nms
            cfg.mask_proto_debug = args.mask_proto_debug
            return YolactSingleModelInference(net,args, dataset)

    def prep_display(self, dets_out, img, h, w, undo_transform=True,binarize_masks=True):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            t = postprocess(dets_out, w, h, visualize_lincomb=self.args.display_lincomb,
                            crop_masks=self.args.crop,
                            score_threshold=self.args.score_threshold, binarize_masks=binarize_masks)
            torch.cuda.synchronize()

        with timer.env('Copy'):
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][:self.args.top_k]
            classes, scores, boxes = [x[:self.args.top_k].cpu().numpy() for x in t[:3]]
        return img_numpy, img_gpu, masks, classes, scores, boxes

