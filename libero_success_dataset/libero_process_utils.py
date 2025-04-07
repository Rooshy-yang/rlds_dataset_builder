import math
import numpy as np
from PIL import Image


def get_libero_image(obs, resize_size, key="agentview_image"):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[key]
    img = np.flipud(img)
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = Image.fromarray(img)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # resize to size seen at train time
    img = img.convert("RGB")
    return np.array(img)

def apply_center_crop(im, t_h, t_w):
    """
    Source: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/utils/obs_utils.py#L268

    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]

def preprocess_image(img):
    image = Image.fromarray(img)
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), we must multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!

    temp_image = np.array(image)  # (H, W, C)
    crop_scale = 0.9
    sqrt_crop_scale = math.sqrt(crop_scale)
    temp_image_cropped = apply_center_crop(
        temp_image,
        t_h=int(sqrt_crop_scale * temp_image.shape[0]),
        t_w=int(sqrt_crop_scale * temp_image.shape[1]),
    )
    temp_image = Image.fromarray(temp_image_cropped)
    temp_image = temp_image.resize(
        image.size, Image.Resampling.BILINEAR
    )  # IMPORTANT: dlimp uses BILINEAR resize
    image = temp_image
    return image


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
