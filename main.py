import numpy as np
import torch

from lib_utils.utils import get_slices


class ObjectDetector:
    def __init__(self, model_name):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

    def predict(self, images: list[np.ndarray]):
        results = self.model(images).pred
        final_results = [result.cpu().numpy() for result in results]
        return final_results

    def predict_slices(self, image: np.ndarray):
        slices = get_slices(image, 320, 320, 0.2)
        all_res = []
        for (start_y, start_x), slice in slices:
            # import ipdb; ipdb.set_trace()
            res = self.predict(slice)[0]
            res[:, 0] += start_x
            res[:, 1] += start_y
            res[:, 2] += start_x
            res[:, 3] += start_y
            all_res.append(res)
        all_res = np.concatenate(all_res, axis=0)
        return all_res


if __name__ == "__main__":
    import cv2
    img = img = cv2.imread("/home/aivn48/.santapo/sod/tests/small-vehicles1.jpeg")
    detector = ObjectDetector("yolov5s")
    slice_res = detector.predict_slices(img)
    # res = detector.predict(img)