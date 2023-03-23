


def get_slices(image: np.ndarray, slice_width: int, slice_height: int, overlapped_ratio: float) -> list[np.ndarray]:
    ...

def nmm(bboxes: np.ndarray, iou_thres: float):
    ...

def draw_bbox(image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    ...