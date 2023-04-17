import numpy as np
import torch
from retinaface.pre_trained_models import get_model as retinaface_get_model

from lib_utils.utils import get_slices


class ObjectDetector:
    def __init__(self, model_name):
        self.model_name = model_name
        if "yolo" in model_name:
            self.model = torch.hub.load('ultralytics/yolov5', model_name)
        elif "retina" in model_name:
            self.model = retinaface_get_model("resnet50_2020-07-20", max_size=2048)

    def _predict(self, images: list[np.ndarray]):
        if "yolo" in self.model_name:
            result = self.model(images).pred
            import ipdb; ipdb.set_trace()
        elif "retina" in self.model_name:
            tmp = self.model.predict_jsons(images)
            result = [instance["bbox"] + [instance["score"]] + [1.0] for instance in tmp \
                        if instance["score"] > 0]
            result = [torch.tensor(result)] if len(result) else [torch.rand(0, 6)]
        return result

    def predict(self, images: list[np.ndarray]):
        results = self._predict(images)
        final_results = [result.cpu().numpy() for result in results]
        return final_results

    def predict_slices(self, image: np.ndarray):
        slices = get_slices(image, 320, 320, 0.2)
        all_res = []
        for (start_y, start_x), slice in slices:
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
    from lib_utils.utils import draw_bbox
    img = img = cv2.imread("tests/face.jpg")
    detector = ObjectDetector("retinaface")
    slice_res = detector.predict_slices(img)
    # slice_res = detector.predict(img)[0]
    img = draw_bbox(img, slice_res)
    cv2.imwrite("test1.jpg", img)

