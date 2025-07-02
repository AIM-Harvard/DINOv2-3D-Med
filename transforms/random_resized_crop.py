from typing import Union, Tuple

import torch
from monai.transforms import RandScaleCrop, Resize, Transform, MapTransform


class RandomResizedCrop3D(Transform):
    """
    Combines monai's random spatial crop followed by resize to the desired size.

    Modification:
    1. The spatial crop is done with same dimensions for all the axes
    2. Handles cases where the image_size is less than the crop_size by choosing
        the smallest dimension as the random scale.

    """

    def __init__(
        self,
        prob: float = 1,
        size: Union[int, Tuple[int, int, int]] = (50, 50, 50),
        scale: Tuple[float, float] = (0.5, 1.0),
    ):
        """
        Args:
            prob (float): Probability of applying the random resized crop.
            size (Union[int, Tuple[int, int, int]]): Desired output size after resizing.
                If an int is provided, it will be used for all three dimensions.
            scale (List[int]): Specifies the lower and upper bounds for the random area of the crop,
             before resizing. The scale is defined with respect to the area of the original image.
        """
        super().__init__()
        if isinstance(size, int):
            size = (size, size, size)
        self.prob = prob
        self.scale = scale
        self.size = size

    def __call__(self, image):
        if torch.rand(1) < self.prob:
            random_scale = torch.empty(1).uniform_(*self.scale).item()
            rand_cropper = RandScaleCrop(random_scale, random_size=False)
            resizer = Resize(self.size, mode="trilinear")

            for transform in [rand_cropper, resizer]:
                image = transform(image)

        return image


class RandomResizedCrop3Dd(MapTransform):
    def __init__(
        self,
        keys,
        prob: float = 1,
        size: Union[int, Tuple[int, int, int]] = (50, 50, 50),
        scale: Tuple[float, float] = (0.5, 1.0),
        allow_missing_keys=False,
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.RandomResizedCrop3D = RandomResizedCrop3D(prob, size, scale)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.RandomResizedCrop3D(d[key])
        return d
