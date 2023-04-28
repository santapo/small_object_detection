import numpy as np
import torch
from retinaface.pre_trained_models import get_model as retinaface_get_model
from torchvision import ops

from lib_utils.utils import get_slices


class ObjectDetector:
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.device = device

        if "yolo" in model_name:
            self.model = torch.hub.load('ultralytics/yolov5', model_name)
            self.model.to(self.device)
        elif "retina" in model_name:
            self.model = retinaface_get_model("resnet50_2020-07-20", max_size=2048, device=self.device)

    def _predict(self, images: list[np.ndarray]):
        if "yolo" in self.model_name:
            result = self.model(images).pred
        elif "retina" in self.model_name:
            tmp = self.model.predict_jsons(images)
            result = [instance["bbox"] + [instance["score"]] + [1.0] for instance in tmp \
                        if instance["score"] > 0]
            result = [torch.tensor(result)] if len(result) else [torch.rand(0, 6)]
        return result

    def predict(self, images: list[np.ndarray]):
        results = self._predict(images)
        final_results = [result.cpu() for result in results]
        return final_results

    def predict_slices(
            self,
            image: np.ndarray,
            slice_width: int,
            slice_height: int,
            overlap_ratio: float
        ):
        slices = get_slices(
            image,
            slice_width,
            slice_height,
            overlap_ratio
        )
        all_res = []
        for (start_y, start_x), slice in slices:
            res = self.predict(slice)[0]
            res[:, 0] += start_x
            res[:, 1] += start_y
            res[:, 2] += start_x
            res[:, 3] += start_y
            all_res.append(res)
        all_res = torch.cat(all_res, dim=0)
        keep_idxs = ops.batched_nms(
            all_res[:, :4],
            all_res[:, 4],
            all_res[:, 5],
            0.1
        )
        res = all_res[keep_idxs]
        return res.numpy()


if __name__ == "__main__":
    import cv2

    from lib_utils.utils import draw_bbox
    img = cv2.imread("tests/small-vehicles1.jpeg")
    detector = ObjectDetector("yolov5s")
    # slice_res = detector.predict_slices(img)
    slice_res = detector.predict(img)[0].numpy()
    img = draw_bbox(img, slice_res)
    cv2.imwrite("test1.jpg", img)

