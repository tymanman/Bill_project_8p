# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""

import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
from PIL import Image

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "ExtentTransform",
    "ResizeTransform",
    "RotationTransform",
    "Rotation3DTransform",
    "ColorTransform",
    "PILColorTransform",
    "MarginCropTransform",
    "Copy_paste_Transform",
]


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside
    the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
        return ret

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])

class Rotation3DTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, anglex, angley, anglez, fov):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        self._set_attributes(locals())
        self.warpR, self.new_shape = self.get_warpR(h, w, anglex, angley, anglez, fov)

    def apply_image(self, img):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        img = cv2.warpPerspective(img, self.warpR, self.new_shape,borderMode=cv2.BORDER_REPLICATE)
        return img

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        keypoint_3dim =  np.concatenate((coords, np.ones((4, 1))), 1).transpose(1,0)
        project_keyp = np.matmul(self.warpR, keypoint_3dim)
        project_keyp[0, :] /= project_keyp[2, :]
        project_keyp[1, :] /= project_keyp[2, :]
        project_keyp = project_keyp[:2, :].transpose(1, 0)
        return project_keyp

    def apply_box(self, box):
        raise NotImplementedError

    def get_warpR(self, h, w, anglex, angley, anglez, fov):
        def rad(x):
            return x * np.pi / 180
        
        # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
        z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
        # 齐次变换矩阵
        rx = np.array([[1, 0, 0, 0],
                    [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                    [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                    [0, 0, 0, 1]], np.float32)
    
        ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                    [0, 1, 0, 0],
                    [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                    [0, 0, 0, 1]], np.float32)
    
        rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                    [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], np.float32)
    
        r = rx.dot(ry).dot(rz)
    
        # 四对点的生成
        pcenter = np.array([w / 2, h / 2, 0, 0], np.float32)
    
        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, h, 0, 0], np.float32) - pcenter
        p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    
        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)
    
        list_dst = [dst1, dst2, dst3, dst4]
    
        org = np.array([[0, 0],
                        [w, 0],
                        [0, h],
                        [w, h]], np.float32)
        dst = np.zeros((4, 2), np.float32)
    
        # 投影至成像平面
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
    
        warpR = cv2.getPerspectiveTransform(org, dst)
        new_shape = (int(dst[:, 0].max() - dst[:, 0].min()), int(dst[:, 1].max() - dst[:, 1].min()))
        offset = (int(dst[:, 0].min()), int(dst[:, 1].min()))
        warpR[0, :] -=  warpR[2, :]*offset[0]
        warpR[1, :] -=  warpR[2, :]*offset[1]
        return warpR, new_shape

class MarginCropTransform(Transform):
    def __init__(self, crop_scale, aug_with_box=True):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, boxes):
        assert boxes.shape[-1] == 2
        w, h = img.shape[1], img.shape[0]
        min_x = boxes[:, 0].min()
        max_x = boxes[:, 0].max()
        min_y = boxes[:, 1].min()
        max_y = boxes[:, 1].max()
        self.crop_left = int(min_x * self.crop_scale)
        self.crop_top = int(min_y * self.crop_scale)
        self.crop_right = int(max_x + (w-1-max_x) * self.crop_scale)
        self.crop_bottom= int(max_y + (h-1-max_y) * self.crop_scale)
        img = img[self.crop_top:self.crop_bottom, self.crop_left:self.crop_right, :]
        return img

    def apply_coords(self, coords):
        coords = coords-np.array([self.crop_left, self.crop_top])
        return coords
    
    def apply_box(self, boxes):
        raise NotImplementedError

class ColorTransform(Transform):
    """
    Generic wrapper for any photometric transforms.
    These transformations should only affect the color space and
        not the coordinate space of the image (e.g. annotation
        coordinates such as bounding boxes should not be changed)
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in an ndarray and returns an ndarray.
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.op(img)

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class PILColorTransform(ColorTransform):
    """
    Generic wrapper for PIL Photometric image transforms,
        which affect the color space and not the coordinate
        space of the image
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in a PIL Image and returns a transformed
                PIL Image.
                For reference on possible operations see:
                - https://pillow.readthedocs.io/en/stable/
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__(op)

    def apply_image(self, img):
        img = Image.fromarray(img)
        return np.asarray(super().apply_image(img))


def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes

class Copy_paste_Transform(Transform):
    def __init__(self, det_img_path, aug_with_box=True):
        super().__init__()
        self._set_attributes(locals())
        self.counter = 0

    def apply_image(self, src_img, boxes):
        assert boxes.shape[-1] == 2
        num_boxes = boxes.shape[0]//4
        self.paste_roi_index = set(np.random.choice(num_boxes, num_boxes,replace=True).tolist())
        roi_boxes = [boxes[index*4:(index+1)*4].astype(np.int64) for index in self.paste_roi_index]
        w, h = src_img.shape[1], src_img.shape[0]
        det_img = cv2.imread(self.det_img_path)
        det_img = cv2.resize(det_img, (w, h))
        roi_mask = np.zeros((h, w, 3)).astype(np.uint8)
        cv2.fillPoly(roi_mask, roi_boxes, (1, 1, 1))
        det_img = cv2.resize(det_img, (src_img.shape[1], src_img.shape[0]))
        aug_img = np.where(roi_mask, src_img, det_img).astype(np.uint8)
        return aug_img

    def apply_coords(self, coords):
        if self.counter in self.paste_roi_index:
            return coords
        else:
            return np.zeros((4, 2))
        self.counter += 1
    
    def apply_box(self, boxes):
        raise NotImplementedError


HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)

# not necessary any more with latest fvcore
NoOpTransform.register_type("rotated_box", lambda t, x: x)
