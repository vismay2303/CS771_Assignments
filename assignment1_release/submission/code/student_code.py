import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import torch
from torch.utils import data
from PIL import Image

from utils import resize_image, load_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly
    (with a probability of 0.5).
    """

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            numpy array: Randomly flipped image
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            return img
        return img

    def __repr__(self):
        return "Random Horizontal Flip"


#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
    """Rescale the input numpy array to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width)

        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
            Pass None during testing: always use CV2.INTER_LINEAR
    """

    def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        # use bilinear if interpolation is not specified
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            numpy array: Rescaled image
        """
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]


        # scale the image
        if isinstance(self.size, int):
            #################################################################################
            # Fill in the code here
            #################################################################################
            h = img.shape[0]
            w = img.shape[1]
            dim = ()
            if h < w:
                dim = (int(self.size * w / h), self.size)
            else:
                dim = (self.size, int(self.size * h / w))
        
            img = resize_image(img, dim, interpolation)
            return img
        else:
            #################################################################################
            # Fill in the code here
            #################################################################################

            img = resize_image(img, self.size, interpolation)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


class RandomSizedCrop(object):
    """Crop the given numpy array to random area and aspect ratio.

    A crop of random area of the original size with a random aspect ratio is made.
    This crop is finally resized to a fixed given size. This is widely used
    as data augmentation for training image classification models.

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        area_range (list of int): range of areas to sample from
        ratio_range (list of int): range of aspect ratios to sample from
        num_trials (int): number of sampling trials
    """

    def __init__(
        self,
        size,
        interpolations=_DEFAULT_INTERPOLATIONS,
        area_range=(0.25, 1.0),
        ratio_range=(0.8, 1.2),
        num_trials=10,
    ):
        self.size = size
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations
        self.num_trials = int(num_trials)
        self.area_range = area_range
        self.ratio_range = ratio_range

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        for attempt in range(self.num_trials):

            # sample target area / aspect ratio from area range and ratio range
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
            
            #################################################################################
            # Fill in the code here
            #################################################################################

            # print(target_area, type(target_area))
            # print(aspect_ratio, type(aspect_ratio))
            
            h = img.shape[0]
            w = img.shape[1]
            # compute the width and height

            # area = w*h, ratio = w/h, w = sqrt(area*ratio), h = sqrt(area/ratio) 
            w1 = np.ceil(np.sqrt(target_area * aspect_ratio)).astype(int)
            h1 = np.ceil(np.sqrt(target_area / aspect_ratio)).astype(int)

            # # note that there are two possibilities
            if h1 <= h and w1 <= w:
                y1 = np.rint(random.uniform(0, (h - h1))).astype(int)
                x1 = np.rint(random.uniform(0, (w - w1))).astype(int)
                y2 = y1 + h1
                x2 = x1 + w1
                img = img[x1:x2,y1:y2]
            else:
                continue


            # crop the image and resize to output size
            if isinstance(self.size, int):
                img = resize_image(img, (self.size, self.size), interpolation)
                return img
            else:
                img = resize_image(img, self.size, interpolation)
                return img

        # Fall back
        if isinstance(self.size, int):
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            #################################################################################
            # Fill in the code here
            #################################################################################
            # with a square sized output, the default is to crop the patch in the center
            # (after all trials fail)
            h = img.shape[0]      
            w = img.shape[1]
            if(w < h):
                img = img[(h - w)//2 : (h + w)//2 , : ]
            else:
                img = img[ : , (w - h)//2:(h + w)//2]
            img = resize_image(img, (self.size, self.size), interpolation)
            return img
            
        else:
            # with a pre-specified output size, the default crop is the image itself
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return (
            "Random Crop"
            + "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
                target_size[0],
                target_size[1],
                self.area_range[0],
                self.area_range[1],
                self.ratio_range[0],
                self.ratio_range[1],
            )
        )


class RandomColor(object):
    """Perturb color channels of a given image
    Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel.
    The sampling is done independently for each channel.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """

    def __init__(self, color_range):
        self.color_range = color_range

    def __call__(self, img):
        #################################################################################
        # Fill in the code here
        #################################################################################

        a_r = random.uniform(-self.color_range, self.color_range)
        a_g = random.uniform(-self.color_range, self.color_range)
        a_b = random.uniform(-self.color_range, self.color_range)

        img[:,:,0] = img[:,:,0] * (1.0 + a_r)
        img[:,:,1] = img[:,:,1] * (1.0 + a_g)
        img[:,:,2] = img[:,:,2] * (1.0 + a_b)

        img = np.clip(img, a_min=0, a_max=255)

        return img

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1 - self.color_range, 1 + self.color_range
        )


class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    Args:
        degree_range (float): range of degree (-d ~ +d)
    """

    def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
        self.degree_range = degree_range
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation
        degree = random.uniform(-self.degree_range, self.degree_range)
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return img

        #################################################################################
        # Fill in the code here

        img = Image.fromarray(img)
        W = img.size[0]
        H = img.size[1]
        img = img.rotate(degree, expand = True)

        #################################################################################
        # get the rectangular with max area in the rotated image

        img = np.asarray(img)

        return img

    def __repr__(self):
        return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range
        )


#################################################################################
# Additional helper functions. No need to modify.
#################################################################################
class ToTensor(object):
    """Convert a ``numpy.ndarray`` image to tensor.
    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # convert image to tensor
        assert (img.ndim > 1) and (img.ndim <= 3)
        if img.ndim == 2:
            img = img[:, :, None]
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        if img.ndim == 3:
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255.0)
        else:
            return tensor_img


class SimpleDataset(data.Dataset):
    """
    A simple dataset using PyTorch dataloader
    """

    def __init__(self, root_folder, file_ext, transforms=None):
        # root folder, split
        self.root_folder = root_folder
        self.transforms = transforms
        self.file_ext = file_ext

        # load all labels
        file_list = glob.glob(os.path.join(root_folder, "*.{:s}".format(file_ext)))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load image and its label (from file name)
        filename = self.file_list[index]
        img = load_image(filename)
        label = os.path.basename(filename)
        label = label.rstrip(".{:s}".format(self.file_ext))
        # apply data augmentations
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
