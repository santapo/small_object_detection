import cv2
import numpy as np
from torchvision import ops

def get_slices_new(image: np.ndarray, slice_width: int, slice_height: int, overlapped_ratio: float) -> list[np.ndarray]:
    h, w = image.shape[:2]
    stepSizeY = int(slice_height - slice_height*overlapped_ratio)
    stepSizeX = int(slice_width - slice_width*overlapped_ratio)
    a = np.arange(0,h-slice_height, stepSizeY)
    a = np.append(a, h-slice_height)
    b = np.arange(0,w-slice_width, stepSizeX)
    b = np.append(b, w-slice_width)
    lst = [((i,j), image[i:i+slice_height, j:j+slice_width, :]) for i in a for j in b]
    return lst

def draw_bbox(image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    img = image.copy()
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    for i in range(bboxes.shape[0]):
        x1 = int(bboxes[i,0])
        y1 = int(bboxes[i,1])
        x2 = int(bboxes[i,2])
        y2 = int(bboxes[i,3])
        text =  str(int(bboxes[i,5])) + ' ' + str(round(bboxes[i,4], 2))
        labelSize = cv2.getTextSize(text, fontFace, fontScale, 1)
        start_point = (x1, y1)
        end_point = (x2, y2)
        img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
        labelSize = cv2.getTextSize(text, fontFace, fontScale, 1)
        img = cv2.rectangle(img, (x1, y1 - labelSize[0][1]), (x1+labelSize[0][0], y1), (60,60,60), cv2.FILLED)
        img = cv2.putText(img, text, start_point, fontFace, fontScale, (0,255,0), thickness=1)
    return img
