import numpy as np
import torch

from utils.utils import get_slices


class ObjectDetector:
    def __init__(self):
        ...

    def predict(self, images: list[np.ndarray]):
        ...

    def predict_slices(self, image: np.ndarray):
        slices = get_slices(image)
        all_res = []
        for (start_y, start_x), slice in slices:
            res = self.predict(slice)
            res[:, 0] += start_y
            res[:, 1] += start_x
            all_res.append(res)
        all_res = np.concatenate(all_res, axis=0)
        return all_res