# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from rectify_roi import *
# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--root")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    root = args.root
    res_dir = "demo_out_ori"
    rectify = True
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    for path in tqdm.tqdm(args.input):
        ori_img = read_image(os.path.join(root, path), format="BGR")
        start_time = time.time()
        height, width = ori_img.shape[:2]
        img = aug.get_transform(ori_img).apply_image(ori_img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        inputs = {"image": img, "height": height, "width": width}
        predictions = model([inputs])[0]["instances"]
        points = predictions.pred_points.cpu().detach().numpy()
        points = points.reshape(-1, 4, 2).astype( np.int32 )
        print(path, points)
        ori_img = ori_img.astype(np.uint8)
        filename = path.split( "/" )[-1]
        if rectify:
            rois_ = [project_rectify(ori_img, point) for point in points]
        cv2.polylines(ori_img, points, True, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA )
        [cv2.putText(ori_img, "score: "+str(score), tuple(points[i][0].tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) for i, score in enumerate(predictions.scores.cpu().detach().numpy().tolist())]
        cv2.imwrite( "{}/{}".format( res_dir, filename ), ori_img )
        for ind, roi in enumerate(rois_):
            cv2.imwrite( "{}/{}".format( res_dir, f"Inst_{ind}_" + filename ), roi )
        print("Done!")
    #     if args.output:
    #         if os.path.isdir(args.output):
    #             assert os.path.isdir(args.output), args.output
    #             out_filename = os.path.join(args.output, os.path.basename(path))
    #         else:
    #             assert len(args.input) == 1, "Please specify a directory with args.output"
    #             out_filename = args.output
    #         visualized_output.save(out_filename)
    #     else:
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    #         if cv2.waitKey(0) == 27:
    #             break  # esc to quit
    # cv2.destroyAllWindows()
