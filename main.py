import numpy as np
import torch

from utils.utils import get_slices


class ObjectDetector:
    def __init__(self, model_name):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

    def predict(self, images: list[np.ndarray]):
        results = self.model(images).pred
        final_results = [result.cpu().numpy() for result in results]
        return final_results

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
