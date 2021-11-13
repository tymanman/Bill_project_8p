# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import json
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ["load_bill_card_instances", "register_bill_card"]


# fmt: off
CLASS_NAMES = (
    "train_ticket", "bank_card"
)
# fmt: on


def load_bill_card_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    meta = []
    with open(os.path.join(dirname, split, split+".sds"), "r") as f:
        anns = f.readlines()
    for line in anns:
        meta.append(json.loads(line.strip()))

    dicts = []
    for ind, item in enumerate(meta):
        file_name = os.path.join(dirname, split, "imgs", item["file_name"])
        height = item["height"]
        width = item["width"]
        r = {
            "file_name": file_name,
            "image_id": ind,
            "height": height,
            "width": width,
        }
        instances = []

        for obj in item["annotations"]:
            keypoint = obj["keypoint"]
            angle = obj["angle"]
            instances.append(
                {"category_id": 0, "bbox": None, "poly": keypoint, "bbox_mode": BoxMode.XYXY_ABS, "angle": angle}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_bill_card(name, dirname, split, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_bill_card_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, split=split
    )
