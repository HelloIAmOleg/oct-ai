import albumentations as A

from cv2 import INTER_CUBIC, BORDER_REFLECT_101

from .utils import prepare_config

DEFAULT_NORM_MEANS = (0.485, 0.456, 0.406)
DEFAULT_NORM_STDS = (0.229, 0.224, 0.225)


class ImageTransformer():

    default_config = {
        "random_state": None,
        "p_augs": 0,
        "rot_angles": (0, 0),
        "rot_p": 1.,
        "normalize": True
    }

    @classmethod
    def get_transforms(cls, config = None, num_images = 1):

        config = prepare_config(cls, config)

        transforms_aug = []

        if config["p_augs"]:
            transforms_aug.append(A.Rotate(limit = config["rot_angles"], interpolation = INTER_CUBIC,
                                           border_mode = BORDER_REFLECT_101, p = config["rot_p"]))
        
        transforms_list = [A.Resize(height = 224, width = 224)]
        if config["normalize"]:
            transforms_list.append(A.Normalize(mean = DEFAULT_NORM_MEANS, std = DEFAULT_NORM_STDS,
                                            max_pixel_value = 255))
        
        # You can fixate `seed` for the Compose - consider doing it later
        if num_images == 1:
            transforms = A.Compose(transforms_aug + transforms_list, seed = config['random_state'])
        else:
            transforms = A.Compose(transforms_aug + transforms_list, additional_targets = {f"image{i}":"image" for i in range(num_images - 1)},
                                   seed = config['random_state'])
        
        return transforms

def apply_transforms(image, transforms):
    return transforms(image = image)['image']