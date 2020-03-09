import torch
from torchvision import transforms
import cv2
import numpy as np
import torch.nn.functional as F

class Rescale(object):
    """Rescale the images in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img1, img2 = sample['image1'], sample['image2']

        assert img1.shape[0] == img2.shape[0] # Height of 2 images should be same
        assert img1.shape[1] == img2.shape[1] # Width of 2 images should be same

        h, w = img1.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img1 = cv2.resize(img1, (new_w, new_h))
        img2 = cv2.resize(img2, (new_w, new_h))

        return {'image1': img1, 'image2': img2}


class RandomRescale(object):
    """Randomly upscales or downscales the images in a sample by a factor uniformly sampled between the
        values specified by [min, max] (Same factor is applied for all the images in a a sample.

    Args:
        min (int): minimum value for the  scale factor (non-negative).
        max (int): maximum value for the  scale factor (non-negative).
    """

    def __init__(self, min=0.0, max=0.0):

        self.min = min
        self.max = max

        assert self.min >= 0  # Scale factor must be non-negative
        assert self.max >= 0  # Scale factor must be non-negative

    def __call__(self, sample):

        factor = np.random.uniform(self.min, self.max)

        img1, img2 = sample['image1'], sample['image2']

        assert img1.shape[0] == img2.shape[0] # Height of 2 images should be same
        assert img1.shape[1] == img2.shape[1] # Width of 2 images should be same

        h, w = img1.shape[:2]

        new_h, new_w = int(h * factor), int(w * factor)

        img1 = cv2.resize(img1, (new_w, new_h))
        img2 = cv2.resize(img2, (new_w, new_h))

        return {'image1': img1, 'image2': img2}


# class RandomSizedCrop(object):
#     """Randomly crop of size 'crop_size' form the images in a smaple. Images will be
#     padded with zeros, if 'Crop_size is bigger than the size of the image
#
#     Args:
#         crop_size (tuple or int): if int a square sized crop would be taken
#     """
#
#     def __init__(self, crop_size):
#
#         assert isinstance(crop_size, (int, tuple))
#
#         if isinstance(crop_size, int):
#             self.crop_size = (crop_size, crop_size)
#         else:
#             self.crop_size = crop_size
#
#     def __call__(self, sample):
#
#         for key, value in sample.items():
#             h, w = sample[key].shape[:2]
#             break
#
#         pad1 = (0,0)
#         pad2 = (0,0)
#
#         if h < self.crop_size[0]:
#             pad1 = (0, self.crop_size[0] - h + 1)
#             h = self.crop_size[0] + 1
#         if w < self.crop_size[1]:
#             pad2 = (0, self.crop_size[1] - w + 1)
#             w = self.crop_size[1] + 1
#
#         sample_aug = {}
#
#         for key, value in sample.items():
#             img = sample[key]
#
#             img = np.pad(img, (pad1, pad2, (0, 0)))
#
#             start_x = np.random.randint(0, w - self.crop_size[1])
#             start_y = np.random.randint(0, h - self.crop_size[0])
#
#             img = img[start_y:start_y + self.crop_size[0],
#                       start_x:start_x + self.crop_size[1], :]
#
#             sample_aug[key] = img
#
#         return sample_aug


# class RandomHorizontalFlip(object):
#
#     def __call__(self, sample):
#
#         flip = np.random.uniform(0, 1)
#
#         if flip > 0.5:
#             sample_aug = {}
#
#             for key, value in sample.items():
#                 sample_aug[key] = np.flip(value, 1)
#
#             return sample_aug
#
#         else:
#             return sample
#
class RandomSizedCrop(object):
    """Randomly crop of size 'crop_size' form the images in a smaple. Images will be
    padded with zeros, if 'Crop_size is bigger than the size of the image

    Args:
        crop_size (tuple or int): if int a square sized crop would be taken
    """

    def __init__(self, crop_size):

        assert isinstance(crop_size, (int, tuple))

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, sample):

        padh = (0,0)
        padw = (0,0)

        h, w = sample.shape[-2:]

        if h < self.crop_size[0]:
            padh = (0, self.crop_size[0] - h + 1)
            h = self.crop_size[0] + 1
        if w < self.crop_size[1]:
            padw = (0, self.crop_size[1] - w + 1)
            w = self.crop_size[1] + 1

        pad = padw + padh
        sample = F.pad(sample, pad)

        start_x = np.random.randint(0, w - self.crop_size[1] + 1)
        start_y = np.random.randint(0, h - self.crop_size[0] + 1)

        sample = sample[:, :, start_y:start_y + self.crop_size[0],
              start_x:start_x + self.crop_size[1]]

        return sample


class RandomHorizontalFlip(object):

    def __call__(self, sample):

        flip = np.random.uniform(0, 1)

        if flip > 0.5:
            return torch.flip(sample, dims=[-1])
        else:
            return sample


# class RandomAffine(object):
#
#     def __init__(self, max_translation_x=0.0, max_translation_y=0.0,
#                         max_rotation=0.0, min_scale=1.0, max_scale=1.0):
#
#         self.max_translation_x = max_translation_x
#         self.max_translation_y = max_translation_y
#         self.max_rotation = max_rotation
#         self.min_scale = min_scale
#         self.max_scale = max_scale
#
#     def __call__(self, sample):
#
#         num_data = len(sample)
#
#         tx = np.random.uniform(-self.max_translation_x, self.max_translation_x)
#         ty = np.random.uniform(-self.max_translation_y, self.max_translation_y)
#         rot = np.random.uniform(-self.max_rotation, self.max_rotation) * np.pi / 180
#         scale = np.random.uniform(self.min_scale, self.max_scale)
#         scale_local = np.hstack([np.random.uniform(self.min_scale, self.max_scale, num_data - 1), 1.0])
#
#         M = np.array([[scale * np.cos(rot), -scale * np.sin(rot), tx],
#                       [scale * np.sin(rot), scale * np.cos(rot), ty]]).astype(np.float32)
#
#         sample_aug = {}
#
#         for key, value in sample.items():
#             img = sample[key]
#             img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
#
#             sample_aug[key] = img
#
#         return sample_aug


class RandomAffine(object):

    def __init__(self, input_size, max_translation_x=0.0, max_translation_y=0.0,
                        max_rotation=0.0, min_scale=1.0, max_scale=1.0):

        self.max_translation_x = max_translation_x
        self.max_translation_y = max_translation_y
        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.height = input_size[0]
        self.width = input_size[1]

    def __call__(self, sample):

        tx = np.random.uniform(-self.max_translation_x, self.max_translation_x)
        ty = np.random.uniform(-self.max_translation_y, self.max_translation_y)
        rot = np.random.uniform(-self.max_rotation, self.max_rotation) * np.pi / 180
        scale = np.random.uniform(self.min_scale, self.max_scale)

        M = torch.tensor([[scale * np.cos(rot), -scale * np.sin(rot), tx],
                      [scale * np.sin(rot), scale * np.cos(rot), ty]], dtype=torch.float32)

        y, x = torch.meshgrid([torch.linspace(-1, 1, self.height),
                               torch.linspace(-1, 1, self.width)])

        y = y.reshape(1, -1)
        x = x.reshape(1, -1)
        ones = torch.ones(1, self.height * self.width)
        grid = torch.cat([x, y, ones], dim=0)

        new_grid = torch.matmul(M, grid).view(1, 2, self.height, self.width).permute(0, 2, 3, 1)

        input_warp = F.grid_sample(sample, new_grid, padding_mode='zeros')

        return input_warp


# class RandomPhotometric(object):
#
#     def __init__(self, noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
#                        brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
#                        min_gamma=1.0, max_gamma=1.0):
#
#         self.noise_stddev = noise_stddev
#         self.min_contrast = min_contrast
#         self.max_contrast = max_contrast
#         self.brightness_stddev = brightness_stddev
#         self.min_colour = min_colour
#         self.max_colour = max_colour
#         self.min_gamma = min_gamma
#         self.max_gamma = max_gamma
#
#     def __call__(self, sample):
#
#         num_data = len(sample)
#         noise = self.noise_stddev * np.random.randn(num_data)
#         brightness = self.brightness_stddev * np.random.randn(num_data)
#         contrast = np.random.uniform(self.min_contrast, self.max_contrast, num_data)
#         gamma = np.random.uniform(self.max_gamma, self.max_gamma, num_data)
#         color = np.random.uniform(self.min_colour, self.max_colour, 3 * num_data)
#
#         gamma_inv = 1 / gamma
#
#         sample_aug = {}
#
#         i = 0
#         for key, value in sample.items():
#             img = sample[key]
#             img = ((img * (1 + contrast[i]) / 255 + brightness[i]) * color[3*i:3*i+3]).clip(0.0, 1.0)
#             img = ((img) ** gamma_inv[i] + noise[i]).clip(0.0, 1.0) * 255
#
#             sample_aug[key] = img.astype(np.uint8)
#             i = i+1
#
#         return sample_aug
#

class RandomPhotometric(object):

    def __init__(self, noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                       brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                       min_gamma=1.0, max_gamma=1.0, num_images=2):

        self.noise_stddev = noise_stddev
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.brightness_stddev = brightness_stddev
        self.min_colour = min_colour
        self.max_colour = max_colour
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.num_images = num_images

    def __call__(self, sample):

        noise = self.noise_stddev * torch.randn(self.num_images)
        brightness = self.brightness_stddev * torch.randn(self.num_images)
        contrast = torch.FloatTensor(self.num_images).uniform_(self.min_contrast, self.max_contrast)
        gamma = torch.FloatTensor(self.num_images).uniform_(self.min_gamma, self.max_gamma)
        color = torch.FloatTensor(3, self.num_images).uniform_(self.min_colour, self.max_colour)
        gamma_inv = 1 / gamma

        sample = sample.permute(2, 3, 1, 0)

        sample = ((sample * (1 + contrast) / 255 + brightness) * color).clamp_(0.0, 1.0)
        sample = (torch.pow(sample, gamma_inv) + noise).clamp_(0.0, 1.0) * 255

        sample = sample.permute(3, 2, 0, 1)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        sample = torch.FloatTensor(sample).permute(0, 3, 1, 2).contiguous()

        return sample












